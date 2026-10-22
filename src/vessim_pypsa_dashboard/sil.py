"""Software-in-the-loop measurement.

A real Python process is launched for every simulation step.  ``os.wait4``
returns the CPU time that process actually consumed, and that CPU time is
turned into watts with a per-core power model.  The controller also *paces* the
simulation: because it blocks until the child exits, one simulated second takes
one real second, which is what makes the case a genuine software-in-the-loop
experiment rather than a trace replay.

The power model is deliberately simple and explicit::

    P = PUE x (base_w + core_w x CPU_seconds / step_seconds)

``base_w`` is the idle IT load (motherboard, memory, fans), ``core_w`` the
per-core power at 100 % utilisation and ``PUE`` the data-centre overhead.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field

import vessim as vs

WORKLOAD = (
    "import time\n"
    "deadline = time.perf_counter() + float({duration})\n"
    "x = 0.0\n"
    "while time.perf_counter() < deadline:\n"
    "    for i in range(20000):\n"
    "        x += (i * 0.5) ** 0.5\n"
)


@dataclass
class SilSample:
    step: int
    cpu_seconds: float
    wall_seconds: float
    watts: float


@dataclass
class SoftwareInTheLoopMonitor(vs.Controller):
    """Measure a real workload and feed its power into a Vessim signal."""

    signal: vs.StaticSignal
    step_s: float = 1.0
    base_w: float = 300.0
    core_w: float = 12.0
    pue: float = 1.2
    load_scale: float = 1.0
    samples: list[dict] = field(default_factory=list)
    _proc: subprocess.Popen | None = None
    _step: int = 0
    _t0: float = 0.0

    def __post_init__(self):
        self._token = uuid.uuid4().hex[:8]

    # -- process handling --------------------------------------------------- #
    def _launch(self) -> None:
        code = WORKLOAD.format(duration=float(self.step_s))
        self._proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._t0 = time.perf_counter()

    def _join(self) -> tuple[float, float]:
        if self._proc is None:
            return 0.0, 0.0
        while True:
            try:
                _, _, usage = os.wait4(self._proc.pid, 0)
                break
            except InterruptedError:  # pragma: no cover - rare
                continue
            except ChildProcessError:  # pragma: no cover - already reaped
                return 0.0, time.perf_counter() - self._t0
        wall = time.perf_counter() - self._t0
        self._proc = None
        return usage.ru_utime + usage.ru_stime, wall

    # -- Controller API ----------------------------------------------------- #
    def start(self, environment) -> None:  # noqa: D401 - Vessim API
        self._launch()

    def step(self, now, microgrid_states) -> None:  # noqa: D401 - Vessim API
        cpu_s, wall_s = self._join()
        utilisation = max(min(cpu_s / max(self.step_s, 1e-9), 1.5), 0.0)
        watts = self.pue * (self.base_w + self.core_w * utilisation) * self.load_scale
        self.signal.set_value(watts)
        self._step += 1
        self.samples.append({
            "step": self._step,
            "cpu_seconds": cpu_s,
            "wall_seconds": wall_s,
            "utilisation": utilisation,
            "watts": watts,
        })
        self._launch()

    def finalize(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=1)
            except Exception:  # pragma: no cover
                pass
            self._proc = None
