"""
PromptRegistry — loads prompts by name and version from disk.

Prompt files are plain text with a YAML front-matter block:

    ---
    name: ner
    version: 1.2.0
    description: ...
    ---
    <prompt body>

Usage:
    registry = PromptRegistry(Path("prompts"))
    template = registry.get("ner")           # latest version
    template = registry.get("ner", "1.0.0")  # exact version
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
import yaml
from packaging.version import Version

logger = structlog.get_logger()


@dataclass(frozen=True)
class PromptEntry:
    name: str
    version: str
    description: str
    body: str


class PromptRegistry:
    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir
        self._index: dict[str, dict[str, PromptEntry]] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str, version: str | None = None) -> str:
        """
        Return the prompt body for the given name and version.
        If version is None, the latest (highest semver) is returned.
        Raises KeyError if name or version is not found.
        """
        versions = self._index.get(name)
        if not versions:
            raise KeyError(f"No prompt found with name '{name}'.")

        if version is None:
            version = self._latest_version(versions)

        entry = versions.get(version)
        if entry is None:
            available = sorted(versions.keys())
            raise KeyError(
                f"Prompt '{name}' version '{version}' not found. "
                f"Available: {available}"
            )

        return entry.body

    def list_versions(self, name: str) -> list[str]:
        """Return all available versions for a prompt name, sorted ascending."""
        versions = self._index.get(name)
        if not versions:
            raise KeyError(f"No prompt found with name '{name}'.")
        return sorted(versions.keys(), key=Version)

    def list_names(self) -> list[str]:
        """Return all registered prompt names."""
        return sorted(self._index.keys())

    def metadata(self, name: str, version: str | None = None) -> PromptEntry:
        """Return the full PromptEntry (including description) for a given name/version."""
        versions = self._index.get(name)
        if not versions:
            raise KeyError(f"No prompt found with name '{name}'.")
        if version is None:
            version = self._latest_version(versions)
        entry = versions.get(version)
        if entry is None:
            raise KeyError(f"Prompt '{name}' version '{version}' not found.")
        return entry

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        for path in sorted(self._dir.glob("*.txt")):
            entry = self._parse_file(path)
            if entry is None:
                continue
            self._index.setdefault(entry.name, {})[entry.version] = entry

    def _parse_file(self, path: Path) -> PromptEntry | None:
        text = path.read_text(encoding="utf-8")

        if not text.startswith("---"):
            return None  # no front-matter — skip silently

        # Split on the closing ---
        parts = text.split("---", maxsplit=2)
        # parts[0] == '' (before opening ---), parts[1] == YAML, parts[2] == body
        if len(parts) < 3:
            return None

        try:
            meta = yaml.safe_load(parts[1])
        except yaml.YAMLError as e:
            logger.warning("Skipping %s: invalid YAML front-matter — %s", path.name, e)
            return None

        if not isinstance(meta, dict):
            logger.warning("Skipping %s: front-matter is not a YAML mapping.", path.name)
            return None

        name = meta.get("name")
        version = meta.get("version")
        if not name or not version:
            logger.warning("Skipping %s: front-matter missing 'name' or 'version'.", path.name)
            return None

        return PromptEntry(
            name=str(name),
            version=str(version),
            description=str(meta.get("description", "")),
            body=parts[2].lstrip("\n"),
        )

    @staticmethod
    def _latest_version(versions: dict[str, PromptEntry]) -> str:
        return max(versions.keys(), key=Version)
