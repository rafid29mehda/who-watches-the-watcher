from collections import namedtuple
from pathlib import Path

import pytest

from wwtw import resources
from wwtw.resources import ResourceError, check_disk, dir_size_mb, doctor_report, wait_for_ram

Usage = namedtuple("Usage", "total used free")
GB = 1024**3


def test_check_disk_raises_when_low(monkeypatch):
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda p: Usage(100 * GB, 95 * GB, 5 * GB))
    with pytest.raises(ResourceError, match="free disk"):
        check_disk(10)


def test_check_disk_ok(monkeypatch):
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda p: Usage(100 * GB, 50 * GB, 50 * GB))
    check_disk(10)


def test_wait_for_ram_returns_when_available(monkeypatch):
    monkeypatch.setattr(resources, "available_ram_gb", lambda: 8.0)
    wait_for_ram(3, max_wait_s=60, sleep=lambda s: pytest.fail("should not sleep"))


def test_wait_for_ram_recovers(monkeypatch):
    readings = iter([1.0, 1.5, 4.0])
    monkeypatch.setattr(resources, "available_ram_gb", lambda: next(readings))
    slept = []
    wait_for_ram(3, max_wait_s=60, poll_s=5, sleep=slept.append, now=lambda: 0.0)
    assert slept == [5, 5]


def test_wait_for_ram_gives_up(monkeypatch):
    monkeypatch.setattr(resources, "available_ram_gb", lambda: 1.0)
    clock = iter([0.0, 30.0, 61.0])
    with pytest.raises(ResourceError, match="RAM"):
        wait_for_ram(3, max_wait_s=60, poll_s=5, sleep=lambda s: None, now=lambda: next(clock))


def test_dir_size_and_doctor(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"x" * 2_000_000)
    assert 1.9 < dir_size_mb(tmp_path) < 2.1
    assert dir_size_mb(tmp_path / "missing") == 0.0
    report = doctor_report(tmp_path)
    assert "Free disk" in report and "Available RAM" in report
