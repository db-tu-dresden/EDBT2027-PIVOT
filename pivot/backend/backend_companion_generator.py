from __future__ import annotations

import os

from pivot.backend.backend_emitter import BackendEmitter
from pivot.lang.backends import build_emitter_registry
from pivot.isa.primitive_registry import PrimitiveRegistry, Definition


class BackendCompanionGenerator:
    """Writes a target's companion unit (C header, Rust module …)."""

    def __init__(self) -> None:
        self._emitters: dict[str, BackendEmitter] = build_emitter_registry()

    def generate(
        self,
        target: str,
        output_dir: str,
        prim_reg: PrimitiveRegistry,
        used_definitions: list[Definition] | None = None,
    ) -> str:
        """Write the companion into ``output_dir``: of the definitions a translation
        calls when the target's library ships a subset (so the companion never names
        an extension the source did not use), else of all the target's."""
        if not (emitter := self._emitters.get(target)):
            raise ValueError(f"No backend emitter registered for target '{target}'")
        if emitter.target.emits_only_used_definitions and used_definitions is not None:
            candidates = used_definitions
        else:
            candidates = [d for primitive in prim_reg.mapping.values() for d in primitive.definitions]
        definitions = [d for d in candidates if d.isa == target]
        if not definitions:
            raise ValueError(f"No primitive definitions found for backend target '{target}'")

        os.makedirs(output_dir, exist_ok=True)
        companion_path = os.path.join(output_dir, emitter.target.companion_filename)
        companion = emitter.companion(definitions)
        if companion.collisions:
            raise ValueError(f"{emitter.target.companion_filename}: distinct definitions share the helper "
                             f"names {', '.join(sorted(companion.collisions))}")
        with open(companion_path, "w") as file:
            file.write(companion.text)
        return companion_path
