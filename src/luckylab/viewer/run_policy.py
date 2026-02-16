#!/usr/bin/env python3
"""Standalone MuJoCo viewer for Go2 SAC policy with web-based viser UI.

Open http://localhost:8080 in your browser after starting.

Controls (via web GUI):
  vx / vy / wz sliders   Velocity commands (step 0.5)
  Stop button            Zero all commands
  Pause / Play           Toggle simulation
  Reset                  Reset robot to standing pose
  Slower / Faster        Adjust simulation speed
  FOV slider             Camera field of view

Terminal keyboard (focus the terminal):
  Up / Down      Forward / backward  (vx +/- 0.5)
  Left / Right   Turn left / right   (wz +/- 0.5)
  , / .          Strafe left / right (vy +/- 0.5)
  Space          Stop all commands
  P              Toggle pause
  R              Reset
  - / =          Speed down / up

Dependencies: mujoco, torch, numpy, viser, trimesh
"""

import argparse
import select
import sys
import termios
import time
import tty
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import trimesh
import trimesh.visual
import trimesh.visual.material
import viser
import viser.transforms as vtf
from mujoco import mjtGeom, mjtObj
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SCENE_XML = str(SCRIPT_DIR / "go2" / "scene.xml")
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # src/luckylab/viewer -> luckylab/
CHECKPOINT_DIR = PROJECT_ROOT / "runs" / "go2_velocity_sac" / "checkpoints"

# ---------------------------------------------------------------------------
# Robot configuration (must match training env exactly)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
DEFAULT_JOINT_POS = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)
ACTION_SCALE = np.array(
    [0.3727530387083568, 0.3727530387083568, 0.24850202580557115] * 4,
    dtype=np.float64,
)
FOOT_GEOM_NAMES = ["FL", "FR", "RL", "RR"]

# Sim timing
SIM_DT = 0.005       # 200 Hz physics
DECIMATION = 4        # policy at 50 Hz
CONTROL_DT = SIM_DT * DECIMATION

# PD gains
DEFAULT_KP = 40.0
DEFAULT_KD = 1.0

# Command limits
CMD_LIMITS = {
    "vx": (-2.0, 3.0),
    "vy": (-1.0, 1.0),
    "wz": (-2.0, 2.0),
}

# Speed presets
SPEED_STEPS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# Policy network (64 -> 256 -> 256 -> 256 -> 12, ELU, tanh output)
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(),
            nn.Linear(256, 12),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_latest_checkpoint(ckpt_dir: Path) -> str | None:
    pts = sorted(ckpt_dir.glob("agent_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if pts:
        return str(pts[-1])
    best = ckpt_dir / "best_agent.pt"
    return str(best) if best.exists() else None


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v from world frame into body frame (quaternion w,x,y,z)."""
    w, u = q[0], q[1:4]
    return (w * w - np.dot(u, u)) * v + 2.0 * np.dot(u, v) * u - 2.0 * w * np.cross(u, v)


# ---------------------------------------------------------------------------
# MuJoCo -> trimesh conversion
# ---------------------------------------------------------------------------
def _extract_texture(mj_model, texid):
    """Extract a PIL Image from a MuJoCo texture."""
    w, h = int(mj_model.tex_width[texid]), int(mj_model.tex_height[texid])
    nc = int(mj_model.tex_nchannel[texid])
    raw = mj_model.tex_data[int(mj_model.tex_adr[texid]) : int(mj_model.tex_adr[texid]) + w * h * nc]
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(nc)
    if mode is None:
        return None
    shape = (h, w) if nc == 1 else (h, w, nc)
    # Flip vertically: MuJoCo uses OpenGL convention, GLTF expects top-left origin
    return Image.fromarray(np.flipud(raw.reshape(shape).astype(np.uint8)), mode=mode)


def _apply_geom_visual(mj_model, geom_idx, mesh, uvs=None):
    """Set mesh visual from MuJoCo material/texture or default color."""
    n = len(mesh.vertices)
    matid = mj_model.geom_matid[geom_idx]

    if matid >= 0:
        rgba = mj_model.mat_rgba[matid]
        # Try texture (only meaningful when mesh has UVs)
        if uvs is not None:
            texid = int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
            if texid < 0:
                texid = int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
            if texid >= 0:
                image = _extract_texture(mj_model, texid)
                if image is not None:
                    mat = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=rgba, baseColorTexture=image,
                        metallicFactor=0.0, roughnessFactor=1.0,
                    )
                    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)
                    return
        # Material color, no texture
        rgba_255 = (rgba * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(rgba_255, (n, 1)))
        return

    # No material — default color based on collision vs visual
    is_col = mj_model.geom_contype[geom_idx] != 0 or mj_model.geom_conaffinity[geom_idx] != 0
    color = np.array([204, 102, 102, 128] if is_col else [31, 128, 230, 255], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, (n, 1)))


def _mesh_geom_to_trimesh(mj_model, geom_idx):
    """Convert a MuJoCo mesh geom to trimesh with material/texture."""
    mid = mj_model.geom_dataid[geom_idx]
    va, vc = int(mj_model.mesh_vertadr[mid]), int(mj_model.mesh_vertnum[mid])
    fa, fc = int(mj_model.mesh_faceadr[mid]), int(mj_model.mesh_facenum[mid])

    verts = mj_model.mesh_vert[va : va + vc]
    faces = mj_model.mesh_face[fa : fa + fc]
    uvs = None

    if mj_model.mesh_texcoordnum[mid] > 0:
        tc_adr = int(mj_model.mesh_texcoordadr[mid])
        tc_num = int(mj_model.mesh_texcoordnum[mid])
        texcoords = mj_model.mesh_texcoord[tc_adr : tc_adr + tc_num]
        face_tc = mj_model.mesh_facetexcoord[fa : fa + fc]
        # Duplicate vertices for per-face UV mapping
        verts = verts[faces.flatten()]
        uvs = texcoords[face_tc.flatten()]
        faces = np.arange(fc * 3).reshape(-1, 3)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    _apply_geom_visual(mj_model, geom_idx, mesh, uvs)
    return mesh


def _primitive_geom_to_trimesh(mj_model, geom_id):
    """Create trimesh for a primitive geom type."""
    size = mj_model.geom_size[geom_id]
    gt = mj_model.geom_type[geom_id]
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=mj_model.geom_rgba[geom_id].copy(),
        metallicFactor=0.0, roughnessFactor=1.0,
    )
    if gt == mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
    elif gt == mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif gt == mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    elif gt == mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    elif gt == mjtGeom.mjGEOM_PLANE:
        mesh = trimesh.creation.box((20, 20, 0.01))
    else:
        raise ValueError(f"Unsupported primitive geom type: {gt}")
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def _merge_geoms(mj_model, geom_ids):
    """Merge multiple geoms into a single trimesh with local transforms."""
    meshes = []
    for gid in geom_ids:
        if mj_model.geom_type[gid] == mjtGeom.mjGEOM_MESH:
            mesh = _mesh_geom_to_trimesh(mj_model, gid)
        else:
            mesh = _primitive_geom_to_trimesh(mj_model, gid)
        T = np.eye(4)
        T[:3, :3] = vtf.SO3(mj_model.geom_quat[gid]).as_matrix()
        T[:3, 3] = mj_model.geom_pos[gid]
        mesh.apply_transform(T)
        meshes.append(mesh)
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def _is_fixed_body(mj_model, body_id):
    if mj_model.body_mocapid[body_id] >= 0:
        return False
    return mj_model.body_weldid[body_id] == 0


def _get_body_name(mj_model, body_id):
    name = mujoco.mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
    return name or f"body_{body_id}"


# ---------------------------------------------------------------------------
# Viser scene
# ---------------------------------------------------------------------------
def _build_scene(server, mj_model):
    """Create viser scene geometry. Returns {(body_id, group): handle}."""
    handles = {}
    server.scene.configure_environment_map(environment_intensity=0.8)
    server.scene.add_frame("/fixed_bodies", show_axes=False)

    # Fixed geometry
    fixed_nonplane = {}
    for i in range(mj_model.ngeom):
        bid = mj_model.geom_bodyid[i]
        if not _is_fixed_body(mj_model, bid):
            continue
        if mj_model.geom_type[i] == mjtGeom.mjGEOM_PLANE:
            name = mujoco.mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, i) or f"plane_{i}"
            server.scene.add_grid(
                f"/fixed_bodies/{name}",
                width=2000.0, height=2000.0, infinite_grid=True,
                fade_distance=50.0, shadow_opacity=0.2,
                position=mj_model.geom_pos[i], wxyz=mj_model.geom_quat[i],
            )
        else:
            fixed_nonplane.setdefault(bid, []).append(i)

    for bid, gids in fixed_nonplane.items():
        server.scene.add_mesh_trimesh(
            f"/fixed_bodies/{_get_body_name(mj_model, bid)}",
            _merge_geoms(mj_model, gids),
            cast_shadow=False, receive_shadow=0.2,
            position=mj_model.body(bid).pos, wxyz=mj_model.body(bid).quat,
        )

    # Dynamic body geometry grouped by (body_id, geom_group)
    body_groups = {}
    for i in range(mj_model.ngeom):
        bid = mj_model.geom_bodyid[i]
        if _is_fixed_body(mj_model, bid):
            continue
        body_groups.setdefault((bid, mj_model.geom_group[i]), []).append(i)

    with server.atomic():
        for (bid, group), gids in body_groups.items():
            handle = server.scene.add_batched_meshes_trimesh(
                f"/bodies/{_get_body_name(mj_model, bid)}/group{group}",
                _merge_geoms(mj_model, gids),
                batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]]),
                batched_positions=np.array([[0.0, 0.0, 0.0]]),
                visible=bool(group == 2),
            )
            handles[(bid, group)] = handle

    return handles


def _update_scene(server, handles, xpos, xmat):
    """Update body poses from copied xpos/xmat arrays."""
    xquat = vtf.SO3.from_matrix(xmat.reshape(-1, 3, 3)).wxyz
    with server.atomic():
        for (bid, _), h in handles.items():
            if not h.visible:
                continue
            h.batched_positions = xpos[bid : bid + 1]
            h.batched_wxyzs = xquat[bid : bid + 1]
        server.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Go2 SAC Policy Viser Viewer")
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to .pt file (default: latest in runs/go2_velocity_sac/checkpoints/)",
    )
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=DEFAULT_KD)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = _find_latest_checkpoint(CHECKPOINT_DIR)
        if args.checkpoint is None:
            parser.error(f"No checkpoints in {CHECKPOINT_DIR}. Pass a path explicitly.")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    model.opt.timestep = SIM_DT
    data = mujoco.MjData(model)

    jnt_ids = [mujoco.mj_name2id(model, mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    jnt_qpos = np.array([model.jnt_qposadr[j] for j in jnt_ids])
    jnt_dof = np.array([model.jnt_dofadr[j] for j in jnt_ids])
    foot_gids = np.array([
        mujoco.mj_name2id(model, mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES
    ])

    # Load policy
    policy = PolicyNet()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net_state = {k: v for k, v in ckpt["policy"].items() if k.startswith("net.")}
    policy.load_state_dict(net_state)
    policy.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Mutable state
    prev_action = np.zeros(12, dtype=np.float64)
    command = np.zeros(3, dtype=np.float64)
    foot_air_time = np.zeros(4, dtype=np.float64)
    target_pos = DEFAULT_JOINT_POS.copy()
    sim_step = 0
    paused = False
    speed_multiplier = 1.0

    def reset():
        nonlocal prev_action, foot_air_time, target_pos, sim_step
        mujoco.mj_resetData(model, data)
        key_id = mujoco.mj_name2id(model, mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        prev_action[:] = 0.0
        foot_air_time[:] = 0.0
        target_pos[:] = DEFAULT_JOINT_POS
        sim_step = 0

    reset()

    # Observation builder (64-dim)
    def build_obs():
        quat = data.qpos[3:7]
        base_lin_vel = _quat_rotate_inverse(quat, data.qvel[0:3])
        base_ang_vel = _quat_rotate_inverse(quat, data.qvel[3:6])
        proj_grav = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        joint_pos_rel = data.qpos[jnt_qpos] - DEFAULT_JOINT_POS
        joint_vel = data.qvel[jnt_dof]
        foot_height = data.geom_xpos[foot_gids, 2].copy()

        foot_contact = np.zeros(4, dtype=np.float64)
        foot_forces = np.zeros(4, dtype=np.float64)
        for ci in range(data.ncon):
            c = data.contact[ci]
            for fi, gid in enumerate(foot_gids):
                if c.geom1 == gid or c.geom2 == gid:
                    foot_contact[fi] = 1.0
                    cfrc = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, cfrc)
                    foot_forces[fi] += abs(cfrc[0])

        for fi in range(4):
            if foot_contact[fi] > 0.5:
                foot_air_time[fi] = 0.0
            else:
                foot_air_time[fi] += CONTROL_DT

        foot_forces_log = np.sign(foot_forces) * np.log1p(np.abs(foot_forces))

        return np.concatenate([
            base_lin_vel,       # 0:3
            base_ang_vel,       # 3:6
            proj_grav,          # 6:9
            joint_pos_rel,      # 9:21
            joint_vel,          # 21:33
            prev_action,        # 33:45
            command,            # 45:48
            foot_height,        # 48:52
            foot_air_time,      # 52:56
            foot_contact,       # 56:60
            foot_forces_log,    # 60:64
        ])

    def policy_step():
        nonlocal prev_action, target_pos
        obs_t = torch.from_numpy(build_obs()).float().unsqueeze(0)
        with torch.no_grad():
            action = policy(obs_t).squeeze(0).numpy()
        prev_action[:] = action
        target_pos[:] = action * ACTION_SCALE + DEFAULT_JOINT_POS

    kp, kd = args.kp, args.kd

    def apply_pd():
        pos = data.qpos[jnt_qpos]
        vel = data.qvel[jnt_dof]
        np.clip(kp * (target_pos - pos) - kd * vel,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
                out=data.ctrl[:12])

    # Viser server + scene
    server = viser.ViserServer(label="Go2 Policy Viewer", port=args.port)
    threadpool = ThreadPoolExecutor(max_workers=1)
    scene_update_pending = False

    print("Building viser scene...")
    handles = _build_scene(server, model)
    print(f"Scene ready. Open http://localhost:{args.port}")

    # -- GUI --
    with server.gui.add_folder("Velocity Commands"):
        vx_slider = server.gui.add_slider(
            "vx (forward)", min=CMD_LIMITS["vx"][0], max=CMD_LIMITS["vx"][1],
            step=0.5, initial_value=0.0,
        )
        vy_slider = server.gui.add_slider(
            "vy (strafe)", min=CMD_LIMITS["vy"][0], max=CMD_LIMITS["vy"][1],
            step=0.5, initial_value=0.0,
        )
        wz_slider = server.gui.add_slider(
            "wz (turn)", min=CMD_LIMITS["wz"][0], max=CMD_LIMITS["wz"][1],
            step=0.5, initial_value=0.0,
        )
        stop_btn = server.gui.add_button("Stop", icon=viser.Icon.PLAYER_STOP)

        @stop_btn.on_click
        def _(_):
            vx_slider.value = vy_slider.value = wz_slider.value = 0.0

    with server.gui.add_folder("Simulation"):
        pause_btn = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)

        @pause_btn.on_click
        def _(_):
            nonlocal paused
            paused = not paused
            pause_btn.name = "Play" if paused else "Pause"
            pause_btn.icon = viser.Icon.PLAYER_PLAY if paused else viser.Icon.PLAYER_PAUSE

        reset_btn = server.gui.add_button("Reset", icon=viser.Icon.REFRESH)

        @reset_btn.on_click
        def _(_):
            reset()
            vx_slider.value = vy_slider.value = wz_slider.value = 0.0

        speed_group = server.gui.add_button_group("Speed", ("Slower", "Faster"))

        @speed_group.on_click
        def _(event: viser.GuiEvent):
            nonlocal speed_multiplier
            idx = SPEED_STEPS.index(speed_multiplier) if speed_multiplier in SPEED_STEPS else 3
            if event.target.value == "Slower":
                idx = max(0, idx - 1)
            else:
                idx = min(len(SPEED_STEPS) - 1, idx + 1)
            speed_multiplier = SPEED_STEPS[idx]

    with server.gui.add_folder("Visualization"):
        fov_slider = server.gui.add_slider(
            "FOV", min=20.0, max=150.0, step=5.0, initial_value=90.0,
        )

        @fov_slider.on_update
        def _(_):
            for client in server.get_clients().values():
                client.camera.fov = fov_slider.value

    with server.gui.add_folder("Status"):
        status_html = server.gui.add_html("")

    # -- Terminal keyboard controls --
    def _read_key():
        """Non-blocking read of a single keypress. Returns key name or None."""
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch == '\x1b':  # Escape sequence (arrow keys)
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                return None
            if sys.stdin.read(1) == '[':
                if not select.select([sys.stdin], [], [], 0.05)[0]:
                    return None
                return {'A': 'up', 'B': 'down', 'C': 'right', 'D': 'left'}.get(
                    sys.stdin.read(1))
            return None
        return ch

    def _handle_key(key):
        nonlocal paused, speed_multiplier
        if key == 'up':
            vx_slider.value = min(vx_slider.value + 0.5, CMD_LIMITS["vx"][1])
        elif key == 'down':
            vx_slider.value = max(vx_slider.value - 0.5, CMD_LIMITS["vx"][0])
        elif key == 'left':
            wz_slider.value = min(wz_slider.value + 0.5, CMD_LIMITS["wz"][1])
        elif key == 'right':
            wz_slider.value = max(wz_slider.value - 0.5, CMD_LIMITS["wz"][0])
        elif key == ',':
            vy_slider.value = min(vy_slider.value + 0.5, CMD_LIMITS["vy"][1])
        elif key == '.':
            vy_slider.value = max(vy_slider.value - 0.5, CMD_LIMITS["vy"][0])
        elif key == ' ':
            vx_slider.value = vy_slider.value = wz_slider.value = 0.0
        elif key in ('p', 'P'):
            paused = not paused
            pause_btn.name = "Play" if paused else "Pause"
            pause_btn.icon = viser.Icon.PLAYER_PLAY if paused else viser.Icon.PLAYER_PAUSE
        elif key in ('r', 'R'):
            reset()
            vx_slider.value = vy_slider.value = wz_slider.value = 0.0
        elif key == '-':
            idx = SPEED_STEPS.index(speed_multiplier) if speed_multiplier in SPEED_STEPS else 3
            speed_multiplier = SPEED_STEPS[max(0, idx - 1)]
        elif key in ('=', '+'):
            idx = SPEED_STEPS.index(speed_multiplier) if speed_multiplier in SPEED_STEPS else 3
            speed_multiplier = SPEED_STEPS[min(len(SPEED_STEPS) - 1, idx + 1)]

    # Set terminal to cbreak mode (keys available immediately, Ctrl+C still works)
    _stdin_fd = sys.stdin.fileno()
    _old_term = termios.tcgetattr(_stdin_fd)
    tty.setcbreak(_stdin_fd)

    print("Terminal controls: ↑↓=fwd/back ←→=turn ,/.=strafe Space=stop P=pause R=reset -/==speed")

    # -- Main loop --
    counter = 0
    try:
        while True:
            t0 = time.time()

            # Poll terminal keyboard
            key = _read_key()
            if key:
                _handle_key(key)

            if not paused:
                command[0] = vx_slider.value
                command[1] = vy_slider.value
                command[2] = wz_slider.value
                if sim_step % DECIMATION == 0:
                    policy_step()
                apply_pd()
                mujoco.mj_step(model, data)
                sim_step += 1
            counter += 1

            # Update viser scene (~50 Hz)
            if counter % DECIMATION == 0 and not scene_update_pending:
                scene_update_pending = True
                xp, xm = data.xpos.copy(), data.xmat.copy()

                def _do_update(xp=xp, xm=xm):
                    nonlocal scene_update_pending
                    try:
                        _update_scene(server, handles, xp, xm)
                    finally:
                        scene_update_pending = False

                threadpool.submit(_do_update)

            # Update status (~5 Hz)
            if counter % 40 == 0:
                vel = data.qvel[0:3]
                state = ('<span style="color:#e74c3c;">PAUSED</span>' if paused
                         else '<span style="color:#2ecc71;">Running</span>')
                status_html.content = (
                    '<div style="font-size:0.85em;line-height:1.5;padding:0.25em 0;">'
                    f"<b>State:</b> {state} &nbsp; <b>Speed:</b> {int(speed_multiplier * 100)}%<br/>"
                    f"<b>Cmd:</b> vx={command[0]:+.1f} vy={command[1]:+.1f} wz={command[2]:+.1f}<br/>"
                    f"<b>Act:</b> vx={vel[0]:+.2f} vy={vel[1]:+.2f} wz={data.qvel[5]:+.2f}<br/>"
                    f"<b>Height:</b> {data.qpos[2]:.3f}m &nbsp; <b>Time:</b> {data.time:.1f}s"
                    "</div>"
                )

            # Real-time pacing (adjusted by speed multiplier)
            dt = (SIM_DT / speed_multiplier) if not paused else SIM_DT
            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, _old_term)
        threadpool.shutdown(wait=True)
        server.stop()


if __name__ == "__main__":
    main()
