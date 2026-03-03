"""Non-blocking terminal keyboard reader for velocity command control."""

import os
import select
import sys
import termios
import threading
import tty


class KeyboardController:
    """Reads WASD/QE keys in a background thread to control velocity commands.

    W/S: forward/backward    A/D: strafe left/right    Q/E: turn left/right
    Space: zero all commands  Esc: quit
    """

    def __init__(self, lin_vel_step: float = 0.2, ang_vel_step: float = 0.1):
        self.lin_vel_step = lin_vel_step
        self.ang_vel_step = ang_vel_step
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0
        self._quit = False
        self._lock = threading.Lock()
        self._old_settings = None
        self._thread = None
        self._fd = sys.stdin.fileno()

    def start(self):
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._quit = True
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    @property
    def should_quit(self) -> bool:
        with self._lock:
            return self._quit

    def get_command(self) -> tuple[float, float, float]:
        with self._lock:
            return self.lin_vel_x, self.lin_vel_y, self.ang_vel_z

    def _read_loop(self):
        while not self._quit:
            if select.select([self._fd], [], [], 0.05)[0]:
                data = os.read(self._fd, 1)
                if not data:
                    continue
                ch = data.decode("utf-8", errors="ignore")
                with self._lock:
                    self._handle_key(ch)

    def _handle_key(self, ch: str):
        if ch == "\x1b":  # Esc
            self._quit = True
        elif ch == " ":
            self.lin_vel_x = 0.0
            self.lin_vel_y = 0.0
            self.ang_vel_z = 0.0
        elif ch in ("w", "W"):
            self.lin_vel_x = min(self.lin_vel_x + self.lin_vel_step, 1.0)
        elif ch in ("s", "S"):
            self.lin_vel_x = max(self.lin_vel_x - self.lin_vel_step, -1.0)
        elif ch in ("a", "A"):
            self.lin_vel_y = min(self.lin_vel_y + self.lin_vel_step, 1.0)
        elif ch in ("d", "D"):
            self.lin_vel_y = max(self.lin_vel_y - self.lin_vel_step, -1.0)
        elif ch in ("q", "Q"):
            self.ang_vel_z = min(self.ang_vel_z + self.ang_vel_step, 0.5)
        elif ch in ("e", "E"):
            self.ang_vel_z = max(self.ang_vel_z - self.ang_vel_step, -0.5)
