"""Repository hygiene guards: maintained sources must not depend on workspace/."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _git(*args, input=None):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        input=input,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def test_local_workspace_is_untracked_and_ignored():
    assert not _git("ls-files", "--", "workspace/").strip()
    probes = [
        "workspace/__boundary_check__/script.py",
        "workspace/__boundary_check__/record.md",
        "workspace/__boundary_check__/data.npz",
    ]
    assert _git(
        "check-ignore", "--stdin", input="\n".join(probes) + "\n"
    ).splitlines() == probes
