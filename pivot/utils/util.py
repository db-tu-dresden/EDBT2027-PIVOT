from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Any, Counter

from pivot.isa.intrinsic_registry import get_intrinsic_registry


def norm(path: str | Path) -> str:
    """Return canonical absolute path for stable matching."""
    return os.path.realpath(os.path.abspath(str(path)))


def print_audit_report(
    label: str,
    found: Counter[str],
    translated: Counter[str],
    logger_func: Callable[[str, str], Any],
    is_global: bool = False,
    level: str = "summary",
) -> None:
    """Print an audit report (per-file or global) sorted by remaining count."""
    border_char = "=" if is_global else "-"
    border_len = 60 if is_global else 50
    header = "GLOBAL TRANSLATION AUDIT SUMMARY" if is_global else f"[Audit] {label}"

    logger_func("\n" + border_char * border_len, level)
    logger_func(header, level)
    logger_func(border_char * border_len, level)

    all_names = sorted(set(found.keys()) | set(translated.keys()))

    report_items = []
    total_found = 0
    total_translated = 0
    intrinsic_registry = get_intrinsic_registry()
    group_totals: dict[str, tuple[int, int]] = {}
    for name in all_names:
        y = found[name]
        x = translated[name]
        remaining = y - x
        isa = intrinsic_registry.intrinsic_isa_of(name)
        report_items.append((isa, name, x, y, remaining))
        group_x, group_y = group_totals.get(isa, (0, 0))
        group_totals[isa] = (group_x + x, group_y + y)
        total_found += y
        total_translated += x

    report_items.sort(key=lambda item: (item[0], -item[4], item[1]))

    current_isa = None
    for isa, name, x, y, remaining in report_items:
        if isa != current_isa:
            group_x, group_y = group_totals.get(isa, (0, 0))
            group_remaining = group_y - group_x
            logger_func(f"  [{isa}] {group_x}/{group_y} ({group_remaining} remaining)", level)
            current_isa = isa
        logger_func(f"  {name:30}: {x:3} / {y:3} ({remaining} remaining)", level)
    logger_func(
        f"  {'Total':30}: {total_translated:3} / {total_found:3} ({total_found - total_translated} remaining)",
        level,
    )
    logger_func(border_char * border_len + "\n", level)
