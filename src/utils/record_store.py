from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image


# Marker delimiting YAML-style frontmatter (Claude-Code-style).
_FRONTMATTER_DELIM = "---"


class RecordStore:
    """File-backed agent memory store.

    Records are stored as plain-text files with optional YAML-style frontmatter
    that captures `name`, `description`, `type`, `step_id`, and `created_at`.
    The frontmatter is what powers the structured memory index — without it,
    the brain only sees opaque filenames.

    Records written without metadata (or older runs) still load: missing fields
    fall back to safe defaults.
    """

    def __init__(self, base_dir: str | Path, encoding: str = "utf-8", max_name_len: int = 80) -> None:
        self.base_dir = Path(base_dir)
        self.encoding = encoding or "utf-8"
        self.max_name_len = max_name_len

    # --- public API --------------------------------------------------------

    def save(
        self,
        text: str,
        file_name: str,
        screenshot: Optional[Image.Image] = None,
        step: Optional[int] = None,
        description: Optional[str] = None,
        record_type: str = "info",
    ) -> str:
        """Persist a record. Returns the on-disk filename (sanitized, with .txt).

        If `description` is empty, derive one heuristically from the first
        non-empty line of `text`, falling back to a humanized filename.
        Callers (e.g., the Agent) may pre-resolve a richer description from
        contextual signals like the brain's `next_goal` and pass it in here.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._sanitize_filename(file_name)
        if not safe_name:
            step_label = step if step is not None else "unknown"
            safe_name = f"record_step_{step_label}.txt"
        text_path = self._ensure_unique_path(self.base_dir / safe_name)

        body = text or ""
        resolved_description = (description or "").strip()
        if not resolved_description:
            resolved_description = self.derive_description(body, text_path.stem)

        frontmatter = self._render_frontmatter(
            name=text_path.stem,
            description=resolved_description,
            record_type=record_type,
            step_id=step,
        )
        text_path.write_text(frontmatter + body, encoding=self.encoding)

        if screenshot:
            screenshot.save(text_path.with_suffix(".png"))
        return text_path.name

    @staticmethod
    def derive_description(text: str, file_stem: str, max_chars: int = 140) -> str:
        """Best-effort one-line description from recorded content.

        Strategy:
          1. First non-empty stripped line of the body, truncated.
          2. Humanized filename if the body has nothing usable.
        """
        if text:
            for raw_line in text.splitlines():
                line = raw_line.strip().lstrip("#-*•> ").strip()
                if line:
                    if len(line) > max_chars:
                        line = line[: max_chars - 1].rstrip() + "…"
                    return line
        # Filename fallback: turn `search_results-2` → `search results 2`
        humanized = re.sub(r"[_\-]+", " ", file_stem or "").strip()
        return humanized or "(no description)"

    def read_files(self, file_names: list[str]) -> str:
        """Return concatenated body text of the requested records (no frontmatter)."""
        if not file_names:
            return "No files requested."
        base_dir = self.base_dir.resolve()
        contents = []
        for raw_name in file_names:
            name = (raw_name or "").strip()
            if not name:
                continue
            file_path = self._resolve_record_path(name, base_dir)
            if not file_path:
                contents.append(f"FILE: {name}\n[Not found]")
                continue
            try:
                full_text = file_path.read_text(encoding=self.encoding)
            except Exception as e:
                contents.append(f"FILE: {file_path.name}\n[Read error: {e}]")
                continue
            _, body = self._split_frontmatter(full_text)
            contents.append(f"FILE: {file_path.name}\n{body}")
        if not contents:
            return "No valid files requested."
        return "\n\n".join(contents)

    def read_metadata(self, file_name: str) -> dict[str, Any]:
        """Return frontmatter dict for one record. Empty dict on miss."""
        base_dir = self.base_dir.resolve()
        path = self._resolve_record_path(file_name, base_dir)
        if not path:
            return {}
        try:
            text = path.read_text(encoding=self.encoding)
        except Exception:
            return {}
        meta, _ = self._split_frontmatter(text)
        return meta

    def list_records(self) -> list[dict[str, Any]]:
        """List every record in this store with parsed frontmatter.

        Each entry: `{"file_name", "name", "description", "type", "step_id"}`.
        Used to build the brain-facing memory index.
        """
        if not self.base_dir.exists():
            return []
        entries: list[dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.txt")):
            try:
                text = path.read_text(encoding=self.encoding)
            except Exception:
                continue
            meta, _ = self._split_frontmatter(text)
            entries.append(
                {
                    "file_name": path.name,
                    "name": meta.get("name") or path.stem,
                    "description": meta.get("description") or "",
                    "type": meta.get("type") or "info",
                    "step_id": self._coerce_int(meta.get("step_id")),
                    "created_at": meta.get("created_at") or "",
                }
            )
        return entries

    # --- internals ---------------------------------------------------------

    def _resolve_record_path(self, name: str, base_dir: Path) -> Optional[Path]:
        candidates = [name]
        if not name.lower().endswith(".txt"):
            candidates.append(f"{name}.txt")
        for candidate in candidates:
            candidate_path = (self.base_dir / candidate).resolve()
            try:
                candidate_path.relative_to(base_dir)
            except ValueError:
                continue
            if candidate_path.exists():
                return candidate_path
        return None

    def _render_frontmatter(
        self,
        name: str,
        description: Optional[str],
        record_type: str,
        step_id: Optional[int],
    ) -> str:
        # Skip frontmatter entirely when there's nothing useful to store — keeps
        # legacy-style files (no metadata) when the caller doesn't supply any.
        has_meta = bool(description) or record_type != "info" or step_id is not None
        if not has_meta:
            return ""
        lines = [
            _FRONTMATTER_DELIM,
            f"name: {self._yaml_escape(name)}",
            f"type: {self._yaml_escape(record_type or 'info')}",
        ]
        if description:
            lines.append(f"description: {self._yaml_escape(description)}")
        if step_id is not None:
            lines.append(f"step_id: {int(step_id)}")
        lines.append(f"created_at: {datetime.utcnow().isoformat(timespec='seconds')}Z")
        lines.append(_FRONTMATTER_DELIM)
        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        """Return (metadata_dict, body_without_frontmatter)."""
        if not text.startswith(_FRONTMATTER_DELIM):
            return {}, text
        lines = text.splitlines()
        if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
            return {}, text
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == _FRONTMATTER_DELIM:
                end_idx = i
                break
        if end_idx is None:
            return {}, text
        meta: dict[str, Any] = {}
        for line in lines[1:end_idx]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
        body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
        return meta, body

    @staticmethod
    def _yaml_escape(value: str) -> str:
        s = str(value).replace("\n", " ").replace("\r", " ").strip()
        if any(ch in s for ch in [":", "#", "'", '"']):
            s = '"' + s.replace('"', '\\"') + '"'
        return s

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _sanitize_filename(self, file_name: str) -> str:
        cleaned = (file_name or "").strip()
        if not cleaned:
            return ""
        for sep in [os.path.sep, os.path.altsep]:
            if sep:
                cleaned = cleaned.replace(sep, "_")
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
        cleaned = cleaned.strip("._-")
        if cleaned and not cleaned.lower().endswith(".txt"):
            cleaned += ".txt"
        return cleaned[: self.max_name_len]

    def _ensure_unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        for i in range(1, 1000):
            candidate = path.with_name(f"{stem}_{i}{suffix}")
            if not candidate.exists():
                return candidate
        return path
