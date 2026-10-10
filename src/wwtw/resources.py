"""Disk/RAM guards so long stages never freeze or fill up the machine."""

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import psutil

from wwtw.config import ResourcesCfg

GB = 1024**3
log = logging.getLogger(__name__)


class ResourceError(RuntimeError):
    """Raised to stop a stage cleanly when the machine is low on disk or RAM."""


def free_disk_gb(path: str | Path = ".") -> float:
    return shutil.disk_usage(path).free / GB


def available_ram_gb() -> float:
    return psutil.virtual_memory().available / GB


def check_disk(min_free_gb: float, path: str | Path = ".") -> None:
    free = free_disk_gb(path)
    if free < min_free_gb:
        raise ResourceError(
            f"Only {free:.1f} GB free disk; need at least {min_free_gb} GB. "
            "Free some space or run `make clean`."
        )


def wait_for_ram(
    min_available_gb: float,
    max_wait_s: float,
    poll_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> None:
    deadline = now() + max_wait_s
    while True:
        avail = available_ram_gb()
        if avail >= min_available_gb:
            return
        if now() >= deadline:
            raise ResourceError(
                f"Available RAM stayed at {avail:.1f} GB (< {min_available_gb} GB) for "
                f"{max_wait_s:.0f}s; stopping cleanly. Close other apps and rerun to resume."
            )
        log.warning(
            "Available RAM %.1f GB < %.1f GB; pausing %.0fs", avail, min_available_gb, poll_s
        )
        sleep(poll_s)


def guard(cfg: ResourcesCfg, path: str | Path = ".") -> None:
    check_disk(cfg.min_free_disk_gb, path)
    wait_for_ram(cfg.min_available_ram_gb, cfg.ram_wait_seconds)


def dir_size_mb(path: str | Path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def doctor_report(root: Path) -> str:
    vm = psutil.virtual_memory()
    load1, _, _ = psutil.getloadavg()
    lines = [
        f"Free disk:      {free_disk_gb(root):.1f} GB",
        f"Available RAM:  {vm.available / GB:.1f} GB of {vm.total / GB:.1f} GB",
        f"CPU load (1m):  {load1:.2f} on {psutil.cpu_count()} cores",
        "Project footprint:",
    ]
    for name in [".venv", "runs", ".cache", ".state", "data", "results"]:
        lines.append(f"  {name:<12} {dir_size_mb(root / name):8.1f} MB")
    ollama = [
        p.info["name"] for p in psutil.process_iter(["name"]) if "ollama" in (p.info["name"] or "")
    ]
    lines.append(f"Ollama processes: {len(ollama)}")
    return "\n".join(lines)
