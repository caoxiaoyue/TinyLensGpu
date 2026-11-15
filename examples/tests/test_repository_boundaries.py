"""Example-suite hygiene guards: maintained sources must not depend on workspace/."""

import ast
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def _git(*args, input=None):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        input=input,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def _sources(suffix):
    names = _git(
        "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    ).split("\0")
    for name in sorted(set(names)):
        path = ROOT / name
        if (
            name
            and not name.startswith("workspace/")
            and "1938" not in name.lower()
            and path.suffix == suffix
            and path.is_file()
        ):
            yield path


def test_python_sources_do_not_depend_on_local_workspace():
    violations = []
    for path in _sources(".py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports = [node.module]
            if any(
                name == "workspace" or name.startswith("workspace.")
                for name in imports
            ):
                violations.append(
                    (str(path.relative_to(ROOT)), node.lineno, "import")
                )
            if (
                path
                in {
                    Path(__file__).resolve(),
                    ROOT / "tests" / "test_repository_boundaries.py",
                }
                or not isinstance(node, ast.Constant)
                or not isinstance(node.value, str)
            ):
                continue
            value = node.value.replace("\\", "/")
            parent = parents.get(node)
            literal_path = value.startswith(
                ("workspace/", "./workspace/", "workspace.")
            ) or f"{ROOT}/workspace/" in value
            path_part = value == "workspace" and (
                isinstance(parent, ast.BinOp)
                and isinstance(parent.op, ast.Div)
                or isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Name)
                and parent.func.id == "Path"
            )
            if literal_path or path_part:
                violations.append(
                    (str(path.relative_to(ROOT)), node.lineno, "path")
                )
    assert not violations, violations


def test_markdown_does_not_link_or_run_local_workspace():
    violations = []
    for path in _sources(".md"):
        text = path.read_text()
        for target in re.findall(r"\]\(([^\s)]+)", text):
            if "://" in target or target.startswith("#"):
                continue
            destination = (path.parent / target.split("#", 1)[0]).resolve()
            if destination.is_relative_to(ROOT / "workspace"):
                violations.append((str(path.relative_to(ROOT)), target))
        for line in text.splitlines():
            if re.search(
                r"\bpython(?:3)?\s+(?:(?:-u|-m)\s+)*workspace[/.]", line
            ):
                violations.append((str(path.relative_to(ROOT)), line))
    assert not violations, violations
