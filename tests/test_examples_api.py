"""Static API checks for runnable code under ``examples/``."""

import ast
import importlib
import re
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
REMOVED_OPERATOR_API = {
    "JointOperatorData",
    "_A_jit_prebound",
    "_joint_A_matvec_jit",
    "as_solver_protocol",
    "build_A_matvec",
    "call_A_matvec",
}


def _example_code_units():
    for path in sorted(EXAMPLES_DIR.rglob("*.py")):
        yield path.relative_to(EXAMPLES_DIR).as_posix(), path.read_bytes()

    python_fence = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
    for path in sorted(EXAMPLES_DIR.rglob("*.md")):
        relative_path = path.relative_to(EXAMPLES_DIR).as_posix()
        for block_index, match in enumerate(
            python_fence.finditer(path.read_text()), start=1
        ):
            yield f"{relative_path}#python-{block_index}", match.group(1)


@pytest.mark.unit
def test_example_python_sources_compile():
    """Every Python source and Markdown Python block should compile."""
    for label, source in _example_code_units():
        compile(source, label, "exec")


@pytest.mark.unit
def test_example_tinylens_imports_resolve():
    """Names imported from TinyLensGpu should exist on their stated surface."""
    failures = []
    for label, source in _example_code_units():
        tree = ast.parse(source, filename=label)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None or not node.module.startswith("TinyLensGpu"):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if alias.name != "*" and not hasattr(module, alias.name):
                    failures.append(
                        f"{label}:{node.lineno}: "
                        f"{node.module}.{alias.name}"
                    )
    assert not failures, "Invalid TinyLensGpu example imports:\n" + "\n".join(
        failures
    )


@pytest.mark.unit
def test_examples_do_not_use_removed_operator_api():
    """Examples should use the typed curvature-operator API exclusively."""
    failures = []
    for label, source in _example_code_units():
        tree = ast.parse(source, filename=label)
        used_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        used_names.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )
        removed_names = sorted(used_names & REMOVED_OPERATOR_API)
        if removed_names:
            failures.append(f"{label}: {', '.join(removed_names)}")
    assert not failures, "Examples use removed operator API:\n" + "\n".join(
        failures
    )
