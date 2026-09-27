import ast
from pathlib import Path

import claimlens

MODELS = Path(claimlens.__file__).parent / "models.py"


def test_models_holds_only_dataclasses():
    """models.py is the typed contracts between layers: dataclasses and
    constants. Logic belongs in the layer that uses it."""
    tree = ast.parse(MODELS.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise AssertionError(f"models.py:{node.lineno} defines {node.name}() — no functions here")
        if isinstance(node, ast.ClassDef):
            decorators = [ast.unparse(d) for d in node.decorator_list]
            assert any(d.startswith("dataclass") for d in decorators), \
                f"models.py:{node.lineno} class {node.name} is not a @dataclass"
