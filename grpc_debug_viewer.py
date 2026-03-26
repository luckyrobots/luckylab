#!/usr/bin/env python3
"""
gRPC Debug Viewer — live camera frames + metadata from LuckyEngine.

Two modes:
  1. Active (default): Drives physics via Session.step() with zero actions.
     Camera frames are returned synchronized with each physics step.
  2. Passive (--passive): Uses CameraService.StreamCamera for read-only
     viewing without driving physics. Scene must already be playing.

Usage (from GrpcPanel "Run Command"):
  uv run --no-sync --group il python ../../../scripts/grpc_debug_viewer.py --cameras Camera --width 256 --height 256

Usage (standalone):
  python scripts/grpc_debug_viewer.py --cameras Camera TopCamera --width 256 --height 256
  python scripts/grpc_debug_viewer.py --passive --cameras Camera --width 640 --height 480 --fps 30

Dependencies: pillow, numpy, grpcio, luckyrobots (all in the luckylab 'il' group)
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
import tkinter as tk
from collections import deque
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk


_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def fmt_vec(values: list[float], max_items: int = 8) -> str:
    if not values:
        return "[]"
    preview = ", ".join(f"{v:+.3f}" for v in values[:max_items])
    suffix = f", ... ({len(values)} total)" if len(values) > max_items else ""
    return f"[{preview}{suffix}]"


def raw_to_rgb_image(data: bytes, width: int, height: int, channels: int) -> Image.Image:
    """Convert raw pixel bytes to a PIL RGB Image."""
    arr = np.frombuffer(data, dtype=np.uint8)

    if channels == 4:
        arr = arr.reshape((height, width, 4))
        return Image.fromarray(arr[:, :, :3], "RGB")
    elif channels == 3:
        arr = arr.reshape((height, width, 3))
        return Image.fromarray(arr, "RGB")
    elif channels == 1:
        arr = arr.reshape((height, width))
        return Image.fromarray(arr, "L").convert("RGB")
    else:
        arr = arr.reshape((height, width, channels))
        return Image.fromarray(arr[:, :, :3], "RGB")


class FpsCounter:
    def __init__(self, window: int = 60):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


class ViewerWindow:
    """Tkinter window that displays camera frames with an FPS overlay."""

    def __init__(self, root: tk.Tk, title: str, width: int, height: int):
        self.window = tk.Toplevel(root)
        self.window.title(title)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.bind("<Key>", self._on_key)

        # Scale up small frames for visibility (minimum 256px on shortest side)
        self._display_scale = max(1, 256 // min(width, height)) if min(width, height) < 256 else 1
        self._dw = width * self._display_scale
        self._dh = height * self._display_scale

        self.canvas = tk.Canvas(self.window, width=self._dw, height=self._dh, bg="black")
        self.canvas.pack()

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        self._closed = False

        # Force window to appear on top
        self.window.lift()
        self.window.attributes("-topmost", True)
        self.window.after(500, lambda: self.window.attributes("-topmost", False))

    def _on_close(self):
        global _shutdown
        _shutdown = True
        self._closed = True

    def _on_key(self, event):
        if event.char == "q":
            global _shutdown
            _shutdown = True
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def update_frame(self, img: Image.Image, fps: float, frame_number: int):
        if self._closed:
            return

        # Draw FPS overlay
        draw = ImageDraw.Draw(img)
        text = f"FPS: {fps:.1f}  Frame: {frame_number}"
        draw.rectangle([(0, 0), (img.width, 18)], fill=(0, 0, 0, 128))
        draw.text((4, 2), text, fill=(0, 255, 0))

        # Scale up if needed
        if self._display_scale > 1:
            img = img.resize((self._dw, self._dh), Image.NEAREST)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self._image_id, image=self._photo)
        self.canvas.update_idletasks()


# ── Active mode: drive physics with zero actions ──────────────────────


def run_active(args: argparse.Namespace, root: tk.Tk) -> int:
    from luckyrobots import Session

    log(f"Active mode: connecting to {args.host}:{args.port} (robot={args.robot})")

    session = Session(host=args.host, port=args.port)
    try:
        session.connect(timeout_s=args.timeout, robot=args.robot)
    except Exception as e:
        log(f"Connection failed: {e}")
        return 1

    # Query schema
    client = session.engine_client
    schema_resp = client.get_agent_schema(agent_name=args.agent)
    schema = schema_resp.schema
    log(f"Agent: {schema.agent_name}")
    log(f"  observation_size: {schema.observation_size}")
    log(f"  action_size:      {schema.action_size}")
    if schema.observation_names:
        log(f"  observation_names: {list(schema.observation_names)[:12]}...")
    if schema.action_names:
        log(f"  action_names:      {list(schema.action_names)[:12]}...")

    action_size = schema.action_size or 12

    # Configure cameras
    cam_cfgs = [
        {"name": name, "width": args.width, "height": args.height}
        for name in args.cameras
    ]
    session.configure_cameras(cam_cfgs)
    log(f"Configured {len(cam_cfgs)} camera(s): {args.cameras} @ {args.width}x{args.height}")

    # Reset
    log("Resetting agent...")
    try:
        obs = session.reset(agent_name=args.agent)
        log(f"Reset OK. frame={obs.frame_number}, cameras={len(obs.camera_frames)}")
        for cf in obs.camera_frames:
            arr = np.frombuffer(cf.data, dtype=np.uint8) if cf.data else np.array([])
            log(
                f"  reset cam='{cf.name}' {cf.width}x{cf.height}x{cf.channels} "
                f"bytes={len(cf.data)} "
                f"min={arr.min() if len(arr) else 'N/A'} "
                f"max={arr.max() if len(arr) else 'N/A'} "
                f"mean={arr.mean():.1f}" if len(arr) else "EMPTY"
            )
    except Exception as e:
        log(f"Reset failed: {e}")
        return 1

    # Create viewer windows
    windows: dict[str, ViewerWindow] = {}
    for name in args.cameras:
        windows[name] = ViewerWindow(root, f"Camera: {name}", args.width, args.height)

    fps_counter = FpsCounter()
    frame_idx = 0
    metadata_interval = max(1, int(args.metadata_hz))

    log("Starting step loop. Press 'q' in any window or close window to quit.")
    log("=" * 60)

    def step_loop():
        nonlocal frame_idx, obs
        global _shutdown

        if _shutdown:
            session.close(stop_engine=False)
            log("Session closed.")
            root.quit()
            return

        # Generate actions: small sinusoidal wiggle or zeros
        if args.wiggle:
            t = time.perf_counter()
            actions = [
                math.sin(t * 2.0 + i * 0.7) * args.wiggle_amp
                for i in range(action_size)
            ]
        else:
            actions = [0.0] * action_size

        try:
            obs = session.step(actions=actions, agent_name=args.agent)
        except Exception as e:
            log(f"Step failed: {e}")
            session.close(stop_engine=False)
            root.quit()
            return

        fps = fps_counter.tick()
        frame_idx += 1

        # Update camera windows
        if frame_idx <= 3 and not obs.camera_frames:
            log(f"  WARNING: step returned 0 camera frames (configured: {args.cameras})")

        for cf in obs.camera_frames:
            # Diagnostic: log first few frames
            if frame_idx <= 3:
                arr = np.frombuffer(cf.data, dtype=np.uint8) if cf.data else np.array([])
                log(
                    f"  cam='{cf.name}' {cf.width}x{cf.height}x{cf.channels} "
                    f"bytes={len(cf.data)} "
                    f"min={arr.min() if len(arr) else 'N/A'} "
                    f"max={arr.max() if len(arr) else 'N/A'} "
                    f"mean={arr.mean():.1f}" if len(arr) else "EMPTY"
                )

            if len(cf.data) == 0:
                continue
            win = windows.get(cf.name)
            if win and not win.closed:
                img = raw_to_rgb_image(cf.data, cf.width, cf.height, cf.channels)
                win.update_frame(img, fps, obs.frame_number)

        # Print metadata periodically
        if frame_idx % metadata_interval == 0:
            log(
                f"frame={obs.frame_number:>6}  "
                f"ts={obs.timestamp_ms}ms  "
                f"fps={fps:>5.1f}  "
                f"obs={fmt_vec(obs.observation)}  "
                f"act={fmt_vec(obs.actions)}  "
                f"cams={len(obs.camera_frames)}"
            )

        # Schedule next step immediately (as fast as physics allows)
        root.after(1, step_loop)

    # Kick off the loop
    root.after(1, step_loop)
    return 0


# ── Passive mode: read-only camera stream ─────────────────────────────


def run_passive(args: argparse.Namespace, root: tk.Tk) -> int:
    import grpc
    from luckyrobots.grpc.generated import camera_pb2, camera_pb2_grpc

    target = f"{args.host}:{args.port}"
    log(f"Passive mode: connecting to {target}")

    channel = grpc.insecure_channel(target)
    stub = camera_pb2_grpc.CameraServiceStub(channel)

    # List cameras if none specified
    camera_names = list(args.cameras)
    if not camera_names:
        try:
            resp = stub.ListCameras(camera_pb2.ListCamerasRequest(), timeout=5.0)
            camera_names = [c.name for c in resp.cameras]
            log(f"Discovered {len(camera_names)} camera(s): {camera_names}")
        except Exception as e:
            log(f"ListCameras failed: {e}")
            return 1

    if not camera_names:
        log("No cameras available.")
        return 1

    # Create viewer windows
    windows: dict[str, ViewerWindow] = {}
    for name in camera_names:
        windows[name] = ViewerWindow(root, f"Camera: {name}", args.width, args.height)

    fps_counters = {name: FpsCounter() for name in camera_names}
    frame_counts = {name: 0 for name in camera_names}

    # Shared state for frames from streaming threads
    _frame_lock = threading.Lock()
    _latest_frames: dict[str, tuple] = {}  # name -> (data, width, height, channels, frame_number, timestamp_ms)
    _stream_errors: dict[str, str] = {}

    def stream_worker(cam_name: str):
        """Background thread: iterate the gRPC stream, stash latest frame."""
        try:
            req = camera_pb2.StreamCameraRequest(
                name=cam_name,
                target_fps=args.fps,
                width=args.width,
                height=args.height,
                format="raw",
            )
            stream = stub.StreamCamera(req, timeout=None)
            for frame in stream:
                if _shutdown:
                    break
                with _frame_lock:
                    _latest_frames[cam_name] = (
                        bytes(frame.data),
                        frame.width,
                        frame.height,
                        frame.channels,
                        frame.frame_number,
                        frame.timestamp_ms,
                    )
        except Exception as e:
            msg = str(e)
            if "StatusCode.CANCELLED" not in msg and "Cancelled" not in msg:
                with _frame_lock:
                    _stream_errors[cam_name] = msg

    # Start stream threads
    threads = []
    for name in camera_names:
        t = threading.Thread(target=stream_worker, args=(name,), daemon=True)
        t.start()
        threads.append(t)
        log(f"Started stream for '{name}' @ {args.width}x{args.height} target_fps={args.fps}")

    log("Streaming. Press 'q' in any window or close window to quit.")
    log("=" * 60)

    metadata_interval = max(1, int(args.metadata_hz))

    def poll_frames():
        global _shutdown

        if _shutdown:
            channel.close()
            log("Channel closed.")
            root.quit()
            return

        with _frame_lock:
            frames = dict(_latest_frames)
            _latest_frames.clear()
            errors = dict(_stream_errors)
            _stream_errors.clear()

        for cam_name, err_msg in errors.items():
            log(f"Stream '{cam_name}' error: {err_msg}")

        for cam_name, (data, w, h, ch, fnum, ts_ms) in frames.items():
            if len(data) == 0:
                continue

            fps = fps_counters[cam_name].tick()
            frame_counts[cam_name] += 1

            # Diagnostic: log pixel stats for first few frames
            if frame_counts[cam_name] <= 3:
                arr = np.frombuffer(data, dtype=np.uint8)
                log(
                    f"  [{cam_name}] DIAG: {w}x{h}x{ch} "
                    f"bytes={len(data)} "
                    f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}"
                )

            win = windows.get(cam_name)
            if win and not win.closed:
                img = raw_to_rgb_image(data, w, h, ch)
                win.update_frame(img, fps, fnum)

            if frame_counts[cam_name] % metadata_interval == 0:
                log(
                    f"[{cam_name}] frame={fnum:>6}  "
                    f"ts={ts_ms}ms  "
                    f"fps={fps:>5.1f}  "
                    f"{w}x{h}x{ch}  "
                    f"bytes={len(data)}"
                )

        root.after(5, poll_frames)

    root.after(5, poll_frames)
    return 0


# ── Entry point ───────────────────────────────────────────────────────


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)

    ap = argparse.ArgumentParser(
        description="gRPC Debug Viewer — live camera frames + metadata from LuckyEngine"
    )
    ap.add_argument("--host", default="127.0.0.1", help="gRPC server host")
    ap.add_argument("--port", type=int, default=50051, help="gRPC server port")
    ap.add_argument("--timeout", type=float, default=30.0, help="Connection timeout (seconds)")

    ap.add_argument(
        "--cameras",
        nargs="+",
        default=["Camera"],
        help="Camera entity name(s) to stream (default: Camera)",
    )
    ap.add_argument("--width", type=int, default=256, help="Requested frame width")
    ap.add_argument("--height", type=int, default=256, help="Requested frame height")

    ap.add_argument("--robot", default="so100", help="Robot name for Session.connect()")
    ap.add_argument("--agent", default="", help="Agent name (empty = default)")

    ap.add_argument(
        "--passive",
        action="store_true",
        help="Passive mode: use CameraService.StreamCamera instead of driving physics",
    )
    ap.add_argument("--fps", type=int, default=30, help="Target FPS for passive stream")

    ap.add_argument(
        "--wiggle",
        action="store_true",
        help="Send small sinusoidal actions to make the robot move (useful for visual confirmation)",
    )
    ap.add_argument(
        "--wiggle-amp",
        type=float,
        default=0.3,
        help="Amplitude of wiggle actions (default: 0.3)",
    )

    ap.add_argument(
        "--metadata-hz",
        type=float,
        default=1.0,
        help="How often to print metadata to terminal (prints every N frames, where N = this value; default 1 = every frame)",
    )

    args = ap.parse_args()

    # Create root Tk window — keep it visible as a small control window
    root = tk.Tk()
    root.title("gRPC Debug Viewer")
    root.geometry("300x60")
    root.bind("<Key>", lambda e: _signal_handler(None, None) if e.char == "q" else None)
    tk.Label(root, text="Press 'q' to quit").pack(pady=10)

    if args.passive:
        rc = run_passive(args, root)
    else:
        rc = run_active(args, root)

    if rc != 0:
        return rc

    try:
        root.mainloop()
    except KeyboardInterrupt:
        log("Interrupted (Ctrl+C)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
