#!/usr/bin/env python3
"""Annotates the intrinsic YAMLs under resources/languages/cpp/intrinsics/{intel,arm}
with PIVOT IR types: each intrinsic whose every parameter resolves gets an `ir:`
block after its `output`, rebuilt on every run.  Change an annotation by changing
a rule here and re-running.

  python scripts/annotate_ir/annotate.py [--base intel|arm] [--dry-run]

Each parameter is typed on its own: a register class gives a float element
(__m256 -> v8f), an integer register takes the element of the name's suffix
(epi32 -> v8i), with the sign the name declares.  Unresolved, and so skipped and
reported: FP16/BF16, MMX __m64, converts and packs with several integer
suffixes, and Helium predicate/count types.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import sys
from pathlib import Path

import yaml

# Make `pivot` importable when run from anywhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pivot.ir.types import IRType, IRKind, parse_ir_token, spell_ir_token, Domain, ElemBits  # noqa: E402
# Annotator-local (offline-only) copy of the positional hint parser.
from hint_registry import IntrinsicTypeHintRegistry  # noqa: E402


# Marker types dumped in compact flow style ({...} / [...]) instead of block.
class _FlowMap(dict):
    pass


class _FlowSeq(list):
    pass


yaml.SafeDumper.add_representer(
    _FlowMap,
    lambda d, data: d.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True),
)
yaml.SafeDumper.add_representer(
    _FlowSeq,
    lambda d, data: d.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True),
)

_INTRINSICS_DIR = os.path.join(_REPO_ROOT, "resources", "languages", "cpp", "intrinsics")

_DOMAIN = {"int": Domain.INT, "uint": Domain.UINT, "float": Domain.FLOAT}

# Positional hints for multi-suffix x86 intrinsics (converts / gather / scatter):
# the name alone cannot say which suffix types which operand.
_HINTS = IntrinsicTypeHintRegistry(
    Path(_INTRINSICS_DIR) / "intrinsic_type_hints_x86.yaml"
)

# Native types whose element the type alone gives.
_NEON_VEC = re.compile(r"^(?P<dom>int|uint|float)(?P<bits>8|16|32|64)x(?P<lanes>\d+)_t$")
_SVE_VEC = re.compile(r"^sv(?P<dom>int|uint|float)(?P<bits>8|16|32|64)_t$")
_SCALAR_INT = re.compile(r"^(?P<sign>u?)int(?P<bits>8|16|32|64)_t$")
_MMASK = re.compile(r"^__mmask(?P<lanes>\d+)$")
# SVE predicate-granularity suffix (`svcntp_b32`, `svptrue_b32`): `_b{N}` with a
# digit, bounded by `_` or end-of-name; not `_b_z` (svnot_b_z, granularity-agnostic).
_SVE_GRANULARITY = re.compile(r"_b(8|16|32|64)(?=_|$)")
# Gather/scatter: the mask predicates the data lanes but takes no domain, since
# the compare producing it may be int or float (see `_mask_ir`).
_GATHER_SCATTER = re.compile(r"(gather|scatter)")
# Operand names that carry the gathered/scattered data (not the index `vindex`).
_RANDOM_ACCESS_DATA_PARAMS = ("a", "src")

# x86 float register classes: native -> (elem_bits, lanes)
_X86_FLOAT = {
    "__m128": (32, 4), "__m256": (32, 8), "__m512": (32, 16),
    "__m128d": (64, 2), "__m256d": (64, 4), "__m512d": (64, 8),
}
# x86 integer register classes: native -> total width in bits
_X86_INT_WIDTH = {"__m128i": 128, "__m256i": 256, "__m512i": 512, "__m64": 64}

# Bare scalar spellings (immediates / counts). Mapped for completeness; whether
# selection constrains on them is a downstream policy question.
_BARE_SCALAR = {
    "int": (Domain.INT, 32), "const int": (Domain.INT, 32),
    "unsigned int": (Domain.UINT, 32), "unsigned": (Domain.UINT, 32),
    "char": (Domain.INT, 8), "signed char": (Domain.INT, 8),
    "unsigned char": (Domain.UINT, 8), "short": (Domain.INT, 16),
    "unsigned short": (Domain.UINT, 16),
    "long long": (Domain.INT, 64), "unsigned long long": (Domain.UINT, 64),
    "__int64": (Domain.INT, 64), "unsigned __int64": (Domain.UINT, 64),
    "constexpr int": (Domain.INT, 32),
    # NEON/SVE scalar float spellings (no (FLOAT,16) token for float16_t)
    "float32_t": (Domain.FLOAT, 32), "float64_t": (Domain.FLOAT, 64),
}

# Immediate-selector enums (x86 rounding/compare/mantissa controls): always a
# compile-time integer, modeled as a plain int32 immediate.
_ENUM_IMM = re.compile(r"^_MM_[A-Z0-9_]+_ENUM$")


def _spell(ir: IRType) -> str | None:
    """Spell an IRType, returning None for element types the IR cannot express
    (e.g. FP16 -> no (FLOAT,16) suffix) -> treated as a gap."""
    return spell_ir_token(ir)


def _type_alone_ir(native: str) -> IRType | None:
    """Element type fully determined by the native type."""
    if native == "float":
        return IRType(IRKind.SCALAR, Domain.FLOAT, ElemBits.BITS_32, 1)
    if native == "double":
        return IRType(IRKind.SCALAR, Domain.FLOAT, ElemBits.BITS_64, 1)
    if m := _SCALAR_INT.match(native):
        dom = Domain.UINT if m.group("sign") == "u" else Domain.INT
        return IRType(IRKind.SCALAR, dom, ElemBits(int(m.group("bits"))), 1)
    if native in _BARE_SCALAR:
        dom, bits = _BARE_SCALAR[native]
        return IRType(IRKind.SCALAR, dom, ElemBits(bits), 1)

    # A mask's element comes from the op, not its type: `_mask_ir` resolves it.
    if native in _X86_FLOAT:
        bits, lanes = _X86_FLOAT[native]
        return IRType(IRKind.VECTOR, Domain.FLOAT, ElemBits(bits), lanes)

    if m := _NEON_VEC.match(native):
        return IRType(IRKind.VECTOR, _DOMAIN[m.group("dom")],
                      ElemBits(int(m.group("bits"))), int(m.group("lanes")))
    if m := _SVE_VEC.match(native):
        return IRType(IRKind.VECTOR, _DOMAIN[m.group("dom")],
                      ElemBits(int(m.group("bits"))), None)  # scalable
    return None


def _int_register_ir(native: str, name: str) -> IRType | None:
    """x86 integer register: element type from the name's suffix,
    or a structureless blob when the name carries no integer element suffix."""
    width = _X86_INT_WIDTH.get(native)
    if width is None:
        return None
    hits = set(re.findall(r"ep([iu])(8|16|32|64)", name))
    if len(hits) == 1:
        sign, bits_s = next(iter(hits))
        bits = int(bits_s)
        dom = Domain.INT if sign == "i" else Domain.UINT
        return IRType(IRKind.VECTOR, dom, ElemBits(bits), width // bits)
    if not hits:  # si128/si256/si512, load/and/or/xor/set: structureless
        return IRType(IRKind.BLOB, total_bits=width)
    return None  # >1 distinct int suffix (convert/pack): ambiguous


def _hint_ir(native: str, position: str, hints: dict) -> str | None:
    """Resolve an x86 integer-register operand from its positional (domain, bits)
    hint, which types each operand of a multi-role name (gather/scatter/convert)
    where the name's single suffix would not (an i32 gather index under `epi64`).

    A width-only hint (domain None, e.g. the `i32` index token) defaults to int;
    the register width fixes the lane count.
    """
    width = _X86_INT_WIDTH.get(native.strip())
    if width is None:
        return None
    hb = hints.get(position)
    if hb is None:
        return None
    domain, elem = hb
    if elem is None:
        return None
    if domain is None:
        domain = Domain.INT  # width-only hint (e.g. the i32 index) -> int default
    return _spell(IRType(IRKind.VECTOR, domain, elem, width // elem.value))


def _resolve_param(native: str, name: str, position: str, hints: dict) -> str | None:
    native = native.strip()
    # SVE reinterpret entries tag the operand with a trailing " op" role marker
    # (e.g. "svint32_t op"); it is not part of the type spelling.
    if native.endswith(" op"):
        native = native[:-len(" op")].strip()
    # Any pointer is an opaque address: the pointee carries no selection signal
    # (target defs spell every pointer as void*).
    if "*" in native:
        return "ptr"
    if _ENUM_IMM.match(native):
        return "int32_t"
    # Types that fully determine themselves (masks, float registers, scalars,
    # NEON/SVE vectors) resolve directly.
    ir = _type_alone_ir(native)
    if ir is not None:
        return _spell(ir)
    # The positional hint tells index from data (gather/scatter/convert), so it
    # wins over the name's single suffix.
    if native in _X86_INT_WIDTH:
        hinted = _hint_ir(native, position, hints)
        if hinted is not None:
            return hinted
    ir = _int_register_ir(native, name)
    return _spell(ir) if ir is not None else None


def _is_mask_native(native: str) -> bool:
    """Whether a native operand type is a predicate/mask register."""
    return bool(_MMASK.match(native)) or native == "svbool_t"


def _mask_elem_from_siblings(
    siblings: list[tuple[bool, str, IRType]], mask_lanes: int | None
) -> tuple[Domain, ElemBits] | None:
    """The `(domain, elem_bits)` a mask predicates, taken from a sibling data
    vector, or ``None`` when the op has no data operand (a pure-mask op).

    A masked op operates on homogeneous data operands, so the mask shares their
    element.  When the mask carries a concrete lane count (x86 ``__mmask{N}``),
    prefer the sibling of matching width, which picks the result element for a
    masked convert (the mask governs the output lanes).  Otherwise (SVE, open
    lanes) prefer an output operand, else the first data operand."""
    data = [
        (is_out, ir)
        for is_out, _pname, ir in siblings
        if ir.kind is IRKind.VECTOR and ir.elem_bits is not None
    ]
    if not data:
        return None
    if mask_lanes is not None:
        for _is_out, ir in data:
            if ir.lanes == mask_lanes:
                return ir.domain, ir.elem_bits
    for is_out, ir in data:
        if is_out:
            return ir.domain, ir.elem_bits
    return data[0][1].domain, data[0][1].elem_bits


def _random_access_mask_elem(
    siblings: list[tuple[bool, str, IRType]], mask_lanes: int | None
) -> tuple[Domain, ElemBits] | None:
    """The `(BITS, elem_bits)` a gather/scatter mask predicates: the width of the
    data operand (``a``/``src``, not ``vindex``), domain-agnostic.  ``None``
    without a data operand."""
    data = [
        ir
        for _is_out, pname, ir in siblings
        if ir.kind is IRKind.VECTOR
        and ir.elem_bits is not None
        and pname in _RANDOM_ACCESS_DATA_PARAMS
    ]
    if not data:
        return None
    chosen = next(
        (ir for ir in data if mask_lanes is None or ir.lanes == mask_lanes),
        data[0],
    )
    return Domain.BITS, chosen.elem_bits


def _governed_lanes(mask_bits: int, siblings: list[tuple[bool, str, IRType]]) -> int:
    """The lane count of an x86 ``__mmask{N}``: N, except for ``__mmask8`` on an op
    with no 8-lane data operand. ``__mmask8`` is the narrowest mask type, so a
    128/256-bit AVX-512VL op on fewer lanes (``_mm_cmplt_epi32_mask``: 4) uses only
    its low bits, and the mask has as many lanes as the narrowest data operand it
    governs. Wider mask types always match their op's lane count."""
    lanes = [ir.lanes for _is_out, _p, ir in siblings
             if ir.kind is IRKind.VECTOR and ir.lanes is not None]
    if mask_bits != 8 or not lanes or mask_bits in lanes:
        return mask_bits
    return min(mask_bits, *lanes)


def _mask_ir(native: str, name: str, siblings: list[tuple[bool, str, IRType]]) -> IRType:
    """Resolve a mask operand to a MASK IRType.

    Lane count: ``__mmask{N}`` -> the governed lanes (x86, baked, see
    `_governed_lanes`); ``svbool_t`` -> None (scalable).
    Predicated element, in priority order:
      0. gather/scatter: the data width, domain-agnostic `_b{N}`;
      1. a sibling data operand's element (concrete `_i`/`_ui`/`_f`, sign kept);
      2. else the name's `_b{N}` granularity suffix (domain-agnostic `_bN`);
      3. else the whole-mask, granularity-agnostic sentinel (`_bALL`)."""
    m = _MMASK.match(native)
    bits = int(m.group("lanes")) if m else None  # svbool_t -> scalable
    lanes = _governed_lanes(bits, siblings) if bits is not None else None

    if _GATHER_SCATTER.search(name):
        if (elem := _random_access_mask_elem(siblings, bits)) is not None:
            return IRType(IRKind.MASK, elem[0], elem[1], lanes)
    if (elem := _mask_elem_from_siblings(siblings, bits)) is not None:
        return IRType(IRKind.MASK, elem[0], elem[1], lanes)
    if gm := _SVE_GRANULARITY.search(name):
        return IRType(IRKind.MASK, Domain.BITS, ElemBits(int(gm.group(1))), lanes)
    return IRType(IRKind.MASK, Domain.BITS, ElemBits.ALL, lanes)


def annotate_entry(entry: dict) -> tuple[dict | None, list[str]]:
    """Return (ir_signature, unresolved); ir_signature is None if incomplete."""
    name = entry.get("name", "")
    inputs = entry.get("input") or {}
    outputs = entry.get("output") or {}
    inputs = inputs if isinstance(inputs, dict) else {}
    outputs = outputs if isinstance(outputs, dict) else {}

    # Positional hints keyed "ret"/"arg0"/"arg1"...; dict order == argument order.
    hints = _HINTS.lookup(name, arg_count=len(inputs)) or {}

    out: dict[str, dict[str, str]] = {"input": {}, "output": {}}
    unresolved: list[str] = []
    positioned = [
        ("input", pname, native, f"arg{i}")
        for i, (pname, native) in enumerate(inputs.items())
    ] + [
        ("output", pname, native, "ret")
        for pname, native in outputs.items()
    ]

    # Pass 1: resolve every non-mask operand, collecting the resolved data-vector
    # IR of each so a mask can borrow its predicated element in pass 2.  A mask's
    # element is a property of the op, not of `__mmask{N}`/`svbool_t` alone, so it
    # cannot be resolved operand-in-isolation.
    mask_slots: list[tuple[str, str, str]] = []  # (section, pname, native)
    siblings: list[tuple[bool, str, IRType]] = []  # (is_output, pname, ir) of data operands
    for section, pname, native, position in positioned:
        # SVE reinterpret entries tag the operand with a trailing " op" role marker.
        clean = str(native).strip()
        if clean.endswith(" op"):
            clean = clean[: -len(" op")].strip()
        if _is_mask_native(clean):
            mask_slots.append((section, pname, clean))
            continue
        tok = _resolve_param(str(native), name, position, hints)
        if tok is None:
            unresolved.append(str(native).strip())
        else:
            out[section][pname] = tok
            if (ir := parse_ir_token(tok)) is not None:
                siblings.append((section == "output", pname, ir))

    # Pass 2: resolve masks with the op's context (siblings + name granularity).
    for section, pname, native in mask_slots:
        tok = _spell(_mask_ir(native, name, siblings))
        if tok is None:
            unresolved.append(native)
        else:
            out[section][pname] = tok

    if unresolved:
        return None, unresolved
    # Restore the original operand order (masks were resolved last but must read
    # positionally, same as the native `input`/`output` maps).
    ordered = {
        "input": {p: out["input"][p] for p in inputs if p in out["input"]},
        "output": {p: out["output"][p] for p in outputs if p in out["output"]},
    }
    return ordered, []


# Files left untouched: Macros.yaml carries hand-written comments and `fold:`
# pseudo-intrinsics (immediate constructors), no operands to IR-type.
_SKIP_FILES = {"Macros.yaml"}


def _flowify(key: str, value):
    """Render input/output maps and flags lists in compact flow style."""
    if key in ("input", "output") and isinstance(value, dict):
        return _FlowMap(value)
    if key == "flags" and isinstance(value, list):
        return _FlowSeq(value)
    return value


def _flow_ir(ir_sig: dict) -> _FlowMap:
    return _FlowMap({
        "input": _FlowMap(ir_sig.get("input") or {}),
        "output": _FlowMap(ir_sig.get("output") or {}),
    })


def _rewrite_entry(entry: dict, ir_sig: dict | None) -> dict:
    """Return a copy of one intrinsic entry with `operation` dropped and a fresh
    `ir` block inserted right after `output` (re-run safe: any stale `ir` is
    rebuilt).  Argument order in `ir` is preserved (never alphabetized); the
    input/output/ir/flags collections are emitted in compact flow style."""
    new_entry: dict = {}
    for key, value in entry.items():
        if key in ("operation", "ir"):
            continue
        new_entry[key] = _flowify(key, value)
        if key == "output" and ir_sig is not None:
            new_entry["ir"] = _flow_ir(ir_sig)
    if ir_sig is not None and "output" not in entry:
        new_entry["ir"] = _flow_ir(ir_sig)
    return new_entry


def run(base: str, *, write: bool) -> None:
    files = sorted(glob.glob(os.path.join(_INTRINSICS_DIR, base, "**", "*.yaml"), recursive=True))
    total = annotated = 0
    unresolved_native = collections.Counter()
    for path in files:
        if os.path.basename(path) in _SKIP_FILES:
            continue
        try:
            docs = yaml.safe_load(open(path))
        except Exception as exc:
            print(f"  ! parse error {path}: {exc}")
            continue
        if not isinstance(docs, list):
            continue

        new_docs = []
        for entry in docs:
            if not isinstance(entry, dict) or "name" not in entry:
                new_docs.append(entry)
                continue
            total += 1
            ir_sig, unresolved = annotate_entry(entry)
            if ir_sig is not None:
                annotated += 1
            else:
                unresolved_native.update(unresolved)
            new_docs.append(_rewrite_entry(entry, ir_sig))

        if write:
            with open(path, "w") as f:
                yaml.safe_dump(new_docs, f, sort_keys=False, default_flow_style=False, width=4096)

    pct = (100.0 * annotated / total) if total else 0.0
    print(f"[{base}] {annotated}/{total} intrinsics annotated in place ({pct:.1f}%)")
    print(f"[{base}] top unresolved native types:")
    for native, n in unresolved_native.most_common(20):
        print(f"    {n:6}  {native}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only, do not rewrite the registry YAMLs")
    ap.add_argument("--base", choices=["intel", "arm"], help="limit to one ISA base")
    ap.add_argument(
        "--intrinsics-dir",
        help="intrinsics root to annotate (default: the cpp corpus).  The native "
        "type spellings are shared across languages, so the same rules re-annotate "
        "e.g. the rust corpus (resources/languages/rust/intrinsics).",
    )
    args = ap.parse_args()
    if args.intrinsics_dir:
        global _INTRINSICS_DIR, _HINTS
        _INTRINSICS_DIR = os.path.abspath(args.intrinsics_dir)
        _HINTS = IntrinsicTypeHintRegistry(
            Path(_INTRINSICS_DIR) / "intrinsic_type_hints_x86.yaml"
        )
    bases = [args.base] if args.base else ["intel", "arm"]
    for base in bases:
        run(base, write=not args.dry_run)
        print()


if __name__ == "__main__":
    main()
