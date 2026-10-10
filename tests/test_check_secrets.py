import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_secrets.sh"


def _repo(tmp_path: Path, content: str) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text(content)
    subprocess.run(["git", "add", "f.txt"], cwd=tmp_path, check=True)
    return tmp_path


def test_flags_api_key(tmp_path):
    fake_key = "sk-" + "A" * 30  # built at runtime so this file never contains the pattern
    repo = _repo(tmp_path, f"key = {fake_key}\n")
    r = subprocess.run(["bash", str(SCRIPT)], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 1
    assert "possible API key" in r.stderr


def test_clean_repo_passes(tmp_path):
    repo = _repo(tmp_path, "nothing secret here\n")
    r = subprocess.run(["bash", str(SCRIPT), "--staged"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0
