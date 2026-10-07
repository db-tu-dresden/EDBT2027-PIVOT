"""Names of the temporaries the preprocessing passes hoist (`pivot_tmp_ret_3`)."""
import re

PIVOT_TEMP_PREFIX = "pivot_tmp_"
# Unanchored, for scanning source text; `is_pivot_temp_name` matches it whole.
TEMP_NAME_PATTERN = rf"{PIVOT_TEMP_PREFIX}[a-z]+_[0-9]+"
_PIVOT_TEMP_RE = re.compile(rf"^{TEMP_NAME_PATTERN}$")


def make_pivot_temp_name(kind: str, index: int) -> str:
    """``make_pivot_temp_name("ret", 3)`` -> ``pivot_tmp_ret_3``."""
    return f"{PIVOT_TEMP_PREFIX}{kind}_{index}"


def is_pivot_temp_name(name: str) -> bool:
    return bool(_PIVOT_TEMP_RE.match(name))
