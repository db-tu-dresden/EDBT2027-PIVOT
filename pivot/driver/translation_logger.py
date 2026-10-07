from __future__ import annotations

from pathlib import Path


class TranslationLogger:
    _LEVELS = {
        "debug": 10,
        "info": 20,
        "summary": 30,
        "warning": 40,
        "error": 50,
    }

    def __init__(self) -> None:
        self._log_path: Path | None = None
        self._messages: list[str] = []
        self._auto_flush = False
        self._min_level_value = self._LEVELS["info"]

    def set_log_path(
        self,
        log_path: str | Path,
        reset: bool = True,
        auto_flush: bool = False,
        min_level: str = "info",
    ) -> None:
        self._log_path = Path(log_path)
        self._auto_flush = auto_flush
        self._min_level_value = self._LEVELS.get(min_level.lower(), self._LEVELS["info"])
        if reset:
            self._messages = []

    def log(self, message: str, level: str = "info", force: bool = False) -> None:
        level_value = self._LEVELS.get(level.lower(), self._LEVELS["info"])
        if not force and level_value < self._min_level_value:
            return
        self._messages.append(message)
        if self._auto_flush:
            self.flush()

    def flush(self) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self._messages)
        if content:
            content += "\n"
        self._log_path.write_text(content, encoding="utf-8")


_cached_logger: TranslationLogger | None = None


def get_translation_logger() -> TranslationLogger:
    global _cached_logger
    if _cached_logger is None:
        _cached_logger = TranslationLogger()
    return _cached_logger
