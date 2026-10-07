"""Evaluate the `fold:` expressions of immediate pseudo-intrinsics
(`_MM_SHUFFLE(1,0,3,2)`, `_CMP_LE_OQ`), declared in the intrinsics YAML.

Every immediate folds to its integer on both sides of matching, so a pattern
with predicate `18` matches only a call whose predicate folds to `18`, and an
immediate left in the output compiles without the source header.
"""

from __future__ import annotations

import ast

# Node types permitted in a `fold:` expression: C integer arithmetic only, so the
# evaluation can never reach names, calls, attributes, or comprehensions.
_FOLD_BINOPS = {
    ast.LShift: lambda a, b: a << b, ast.RShift: lambda a, b: a >> b,
    ast.BitOr: lambda a, b: a | b, ast.BitAnd: lambda a, b: a & b,
    ast.BitXor: lambda a, b: a ^ b, ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
}
_FOLD_UNARYOPS = {ast.Invert: lambda a: ~a, ast.USub: lambda a: -a, ast.UAdd: lambda a: +a}


def _eval_fold_node(node: ast.AST, env: dict[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return _eval_fold_node(node.body, env)
    if isinstance(node, ast.BinOp) and type(node.op) in _FOLD_BINOPS:
        return _FOLD_BINOPS[type(node.op)](
            _eval_fold_node(node.left, env), _eval_fold_node(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _FOLD_UNARYOPS:
        return _FOLD_UNARYOPS[type(node.op)](_eval_fold_node(node.operand, env))
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    raise ValueError(f"disallowed fold expression node: {ast.dump(node)}")


def fold_intrinsic_call(registry, name: str, arg_texts: list[str]) -> str | None:
    """Fold a fold pseudo-intrinsic call (`_MM_SHUFFLE(1,0,3,2)`) to its integer
    value string; None when `name` has no fold or an argument is not constant."""
    fold = registry.fold_of(name)
    if fold is None:
        return None
    param_names, expr = fold
    return eval_fold(expr, param_names, arg_texts)


def fold_constant(registry, name: str) -> str | None:
    """Fold a bare zero-argument immediate macro (`_CMP_LE_OQ`, `_MM_CMPINT_LT`)
    to its integer value string, else None.  It selects an operation, so it binds
    to a pattern by value."""
    return fold_intrinsic_call(registry, name.strip(), [])


def eval_fold(expr: str, param_names: list[str], arg_texts: list[str]) -> str | None:
    """Evaluate a `fold:` expression (`(z<<6)|(y<<4)|...`) with the parameters bound
    to constant integer arguments, returning the resulting integer as a string.

    Returns None when the arguments are not all integer literals (so a fold over a
    runtime value is left untranslated) or the expression uses anything beyond plain
    integer arithmetic. C ``<<``/``|``/``&`` share Python's spelling and precedence.
    """
    if len(param_names) != len(arg_texts):
        return None
    env: dict[str, int] = {}
    for name, text in zip(param_names, arg_texts):
        try:
            env[name] = int(text.strip(), 0)  # 0x.. or decimal
        except (ValueError, AttributeError):
            return None
    try:
        tree = ast.parse(expr, mode="eval")
        return str(_eval_fold_node(tree, env))
    except (ValueError, SyntaxError):
        return None
