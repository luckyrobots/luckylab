#!/usr/bin/env python3
"""
Multi-Camera Stress Test — validates that N concurrent StreamCamera streams
don't starve the main thread.

Connects in passive mode, opens one StreamCamera per camera entity,
and displays all feeds side-by-side with per-camera FPS counters.

The key metric: with the CameraFrameBroker fix, FPS should stay close to
the editor frame rate regardless of camera count.  Without the fix,
adding a second camera tanks from ~20 FPS to <1 FPS.

Usage:
  python grpc_multicam_test.py                          # auto-discover all cameras
  python grpc_multicam_test.py --cameras Camera TopCamera
  python grpc_multicam_test.py --cameras Camera TopCamera --width 640 --height 480

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
from PIL import Image, ImageDraw, ImageTk


_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def raw_to_rgb_image(data: bytes, width: int, height: int, channels: int) -> Image.Image:
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


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)

    ap = argparse.ArgumentParser(description="Multi-camera gRPC stress test")
    ap.add_argument("--host", default="127.0.0.1", help="gRPC server host")
    ap.add_argument("--port", type=int, default=50051, help="gRPC server port")
    ap.add_argument("--cameras", nargs="+", default=[], help="Camera names (empty = auto-discover)")
    ap.add_argument("--width", type=int, default=256, help="Requested frame width")
    ap.add_argument("--height", type=int, default=256, help="Requested frame height")
    ap.add_argument("--fps", type=int, default=30, help="Target FPS per stream")
    ap.add_argument("--duration", type=int, default=0, help="Auto-quit after N seconds (0 = run forever)")

    ap.add_argument("--wiggle", action="store_true", help="Drive physics with sinusoidal actions (requires Session)")
    ap.add_argument("--wiggle-amp", type=float, default=0.3, help="Wiggle amplitude (default: 0.3)")
    ap.add_argument("--robot", default="so100", help="Robot name for Session.connect()")
    ap.add_argument("--agent", default="", help="Agent name (empty = default)")
    ap.add_argument("--timeout", type=float, default=30.0, help="Connection timeout (seconds)")

    args = ap.parse_args()

    import grpc
    from luckyrobots.grpc.generated import camera_pb2, camera_pb2_grpc

    target = f"{args.host}:{args.port}"
    log(f"Connecting to {target}")

    channel = grpc.insecure_channel(target)
    stub = camera_pb2_grpc.CameraServiceStub(channel)

    # Discover cameras
    camera_names = list(args.cameras)
    if not camera_names:
        try:
            resp = stub.ListCameras(camera_pb2.ListCamerasRequest(), timeout=5.0)
            camera_names = [c.name for c in resp.cameras]
            log(f"Auto-discovered {len(camera_names)} camera(s): {camera_names}")
        except Exception as e:
            log(f"ListCameras failed: {e}")
            return 1

    if not camera_names:
        log("No cameras found.")
        return 1

    # ── Optional wiggle: set up Session to drive physics ──
    session = None
    action_size = 12
    if args.wiggle:
        from luckyrobots import Session

        log(f"Wiggle mode: connecting Session (robot={args.robot})")
        session = Session(host=args.host, port=args.port)
        try:
            session.connect(timeout_s=args.timeout, robot=args.robot)
        except Exception as e:
            log(f"Session connect failed: {e}")
            return 1

        schema_resp = session.engine_client.get_agent_schema(agent_name=args.agent)
        action_size = schema_resp.schema.action_size or 12
        log(f"  action_size={action_size}")

        cam_cfgs = [{"name": name, "width": args.width, "height": args.height} for name in camera_names]
        session.configure_cameras(cam_cfgs)

        session.reset(agent_name=args.agent)
        log("Session reset OK, wiggle active.")

    mode = "wiggle + passive streams" if args.wiggle else "passive streams only"
    log(f"Streaming {len(camera_names)} camera(s): {camera_names} @ {args.width}x{args.height} target_fps={args.fps} [{mode}]")
    log("=" * 60)

    # ── Tkinter setup: one window with all cameras side by side ──

    root = tk.Tk()
    root.title(f"Multi-Camera Test ({len(camera_names)} cameras)")
    root.bind("<Key>", lambda e: _signal_handler(None, None) if e.char == "q" else None)
    root.protocol("WM_DELETE_WINDOW", lambda: _signal_handler(None, None))

    display_scale = max(1, 256 // min(args.width, args.height)) if min(args.width, args.height) < 256 else 1
    dw = args.width * display_scale
    dh = args.height * display_scale
    total_w = dw * len(camera_names)

    canvas = tk.Canvas(root, width=total_w, height=dh + 40, bg="black")
    canvas.pack()

    # Per-camera image items on canvas
    cam_image_ids = {}
    cam_photos: dict[str, Optional[ImageTk.PhotoImage]] = {}
    for i, name in enumerate(camera_names):
        x = i * dw
        cam_image_ids[name] = canvas.create_image(x, 0, anchor=tk.NW)
        cam_photos[name] = None
        # Label
        canvas.create_text(x + dw // 2, dh + 10, text=name, fill="white", font=("Consolas", 10))

    # Status bar
    status_id = canvas.create_text(total_w // 2, dh + 30, text="Starting...", fill="lime", font=("Consolas", 9))

    # ── Streaming threads ──

    fps_counters = {name: FpsCounter() for name in camera_names}
    frame_counts = {name: 0 for name in camera_names}

    _frame_lock = threading.Lock()
    _latest_frames: dict[str, tuple] = {}
    _stream_errors: dict[str, str] = {}

    start_time = time.perf_counter()

    def stream_worker(cam_name: str):
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

    for name in camera_names:
        t = threading.Thread(target=stream_worker, args=(name,), daemon=True)
        t.start()

    # ── Poll & render loop ──

    def poll_frames():
        global _shutdown

        if _shutdown:
            if session:
                session.close(stop_engine=False)
            channel.close()
            log("Done.")
            root.quit()
            return

        # Auto-quit after duration
        if args.duration > 0 and (time.perf_counter() - start_time) > args.duration:
            _shutdown = True

        # Drive physics with wiggle actions
        if session:
            t = time.perf_counter()
            actions = [math.sin(t * 2.0 + i * 0.7) * args.wiggle_amp for i in range(action_size)]
            try:
                session.step(actions=actions, agent_name=args.agent)
            except Exception as e:
                log(f"Step failed: {e}")
                _shutdown = True

        with _frame_lock:
            frames = dict(_latest_frames)
            _latest_frames.clear()
            errors = dict(_stream_errors)
            _stream_errors.clear()

        for cam_name, err_msg in errors.items():
            log(f"STREAM ERROR [{cam_name}]: {err_msg}")

        fps_parts = []
        for cam_name, (data, w, h, ch, fnum, ts_ms) in frames.items():
            if len(data) == 0:
                continue

            fps = fps_counters[cam_name].tick()
            frame_counts[cam_name] += 1
            fps_parts.append(f"{cam_name}={fps:.1f}")

            img = raw_to_rgb_image(data, w, h, ch)

            # FPS overlay on each camera
            draw = ImageDraw.Draw(img)
            draw.rectangle([(0, 0), (img.width, 16)], fill=(0, 0, 0, 180))
            draw.text((4, 1), f"{fps:.1f} fps  #{fnum}", fill=(0, 255, 0))

            if display_scale > 1:
                img = img.resize((dw, dh), Image.NEAREST)

            cam_photos[cam_name] = ImageTk.PhotoImage(img)
            canvas.itemconfig(cam_image_ids[cam_name], image=cam_photos[cam_name])

            # Log periodically
            if frame_counts[cam_name] % 60 == 1:
                log(f"[{cam_name}] frame={fnum:>6}  fps={fps:.1f}  {w}x{h}x{ch}")

        # Update status bar
        if fps_parts:
            elapsed = time.perf_counter() - start_time
            canvas.itemconfig(status_id, text=f"FPS: {' | '.join(fps_parts)}   elapsed: {elapsed:.0f}s")

        canvas.update_idletasks()
        root.after(5, poll_frames)

    root.after(100, poll_frames)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        log("Interrupted")

    # Print summary
    log("=" * 60)
    log("SUMMARY:")
    for name in camera_names:
        count = frame_counts[name]
        elapsed = time.perf_counter() - start_time
        avg_fps = count / elapsed if elapsed > 0 else 0
        log(f"  {name}: {count} frames in {elapsed:.1f}s = {avg_fps:.1f} avg fps")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
