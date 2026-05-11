import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_track_d.sh"


def test_script_exists_and_is_executable():
    assert SCRIPT.exists()
    assert os.access(SCRIPT, os.X_OK), "run_track_d.sh must be chmod +x"


def test_script_help_lists_smoke_and_full():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        capture_output=True, text=True,
    )
    assert "smoke" in proc.stdout.lower()
    assert "full" in proc.stdout.lower()


def test_script_rejects_unknown_mode():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "weird"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
