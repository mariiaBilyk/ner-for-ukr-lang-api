"""
Unit tests for PromptRegistry.

All tests use a synthetic in-memory prompt directory built via tmp_path —
no dependency on the real prompts/ folder, so tests are hermetic.
"""

import pytest
from pathlib import Path
from prompts.registry import PromptEntry, PromptRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_prompt(directory: Path, filename: str, name: str, version: str,
                 body: str, description: str = "") -> Path:
    """Write a well-formed prompt file and return its path."""
    desc_line = f'description: "{description}"' if description else ""
    content = f"---\nname: {name}\nversion: {version}\n{desc_line}\n---\n{body}"
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def prompts_dir(tmp_path: Path) -> Path:
    """A temporary directory pre-populated with three prompt versions."""
    write_prompt(tmp_path, "ner-v1.txt",   name="ner", version="1.0.0",
                 body="Prompt body v1.0.0", description="Baseline")
    write_prompt(tmp_path, "ner-v2.txt",   name="ner", version="1.1.0",
                 body="Prompt body v1.1.0", description="Improved")
    write_prompt(tmp_path, "ner-v3.txt",   name="ner", version="2.0.0",
                 body="Prompt body v2.0.0", description="Major rewrite")
    write_prompt(tmp_path, "other.txt",    name="other-prompt", version="1.0.0",
                 body="Other prompt body")
    return tmp_path


@pytest.fixture()
def registry(prompts_dir: Path) -> PromptRegistry:
    return PromptRegistry(prompts_dir)


# ---------------------------------------------------------------------------
# get() — load by name (latest)
# ---------------------------------------------------------------------------

class TestGetLatest:
    def test_returns_body_of_latest_semver(self, registry: PromptRegistry):
        body = registry.get("ner")
        assert body == "Prompt body v2.0.0"

    def test_latest_uses_semver_not_lexicographic(self, tmp_path: Path):
        # 1.10.0 > 1.9.0 semantically; lexicographically "9" > "10"
        write_prompt(tmp_path, "a.txt", name="x", version="1.9.0",  body="nine")
        write_prompt(tmp_path, "b.txt", name="x", version="1.10.0", body="ten")
        registry = PromptRegistry(tmp_path)
        assert registry.get("x") == "ten"

    def test_unknown_name_raises_key_error(self, registry: PromptRegistry):
        with pytest.raises(KeyError, match="does-not-exist"):
            registry.get("does-not-exist")


# ---------------------------------------------------------------------------
# get() — load by explicit version
# ---------------------------------------------------------------------------

class TestGetByVersion:
    def test_returns_correct_body_for_version(self, registry: PromptRegistry):
        assert registry.get("ner", "1.0.0") == "Prompt body v1.0.0"
        assert registry.get("ner", "1.1.0") == "Prompt body v1.1.0"
        assert registry.get("ner", "2.0.0") == "Prompt body v2.0.0"

    def test_unknown_version_raises_key_error(self, registry: PromptRegistry):
        with pytest.raises(KeyError, match="9.9.9"):
            registry.get("ner", "9.9.9")

    def test_error_message_lists_available_versions(self, registry: PromptRegistry):
        with pytest.raises(KeyError, match="1.0.0"):
            registry.get("ner", "9.9.9")


# ---------------------------------------------------------------------------
# list_versions()
# ---------------------------------------------------------------------------

class TestListVersions:
    def test_returns_all_versions_sorted_ascending(self, registry: PromptRegistry):
        assert registry.list_versions("ner") == ["1.0.0", "1.1.0", "2.0.0"]

    def test_semver_order_not_lexicographic(self, tmp_path: Path):
        write_prompt(tmp_path, "a.txt", name="x", version="1.9.0",  body="")
        write_prompt(tmp_path, "b.txt", name="x", version="1.10.0", body="")
        registry = PromptRegistry(tmp_path)
        assert registry.list_versions("x") == ["1.9.0", "1.10.0"]

    def test_unknown_name_raises_key_error(self, registry: PromptRegistry):
        with pytest.raises(KeyError, match="missing"):
            registry.list_versions("missing")


# ---------------------------------------------------------------------------
# list_names()
# ---------------------------------------------------------------------------

class TestListNames:
    def test_returns_all_names_sorted(self, registry: PromptRegistry):
        assert registry.list_names() == ["ner", "other-prompt"]

    def test_empty_directory_returns_empty_list(self, tmp_path: Path):
        registry = PromptRegistry(tmp_path)
        assert registry.list_names() == []


# ---------------------------------------------------------------------------
# metadata()
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_returns_prompt_entry_dataclass(self, registry: PromptRegistry):
        meta = registry.metadata("ner", "1.0.0")
        assert isinstance(meta, PromptEntry)

    def test_entry_fields_match_front_matter(self, registry: PromptRegistry):
        meta = registry.metadata("ner", "1.0.0")
        assert meta.name == "ner"
        assert meta.version == "1.0.0"
        assert meta.description == "Baseline"
        assert meta.body == "Prompt body v1.0.0"

    def test_no_version_returns_latest(self, registry: PromptRegistry):
        meta = registry.metadata("ner")
        assert meta.version == "2.0.0"

    def test_unknown_name_raises_key_error(self, registry: PromptRegistry):
        with pytest.raises(KeyError, match="ghost"):
            registry.metadata("ghost")

    def test_unknown_version_raises_key_error(self, registry: PromptRegistry):
        with pytest.raises(KeyError, match="9.9.9"):
            registry.metadata("ner", "9.9.9")


# ---------------------------------------------------------------------------
# Malformed / edge-case files — registry must stay resilient
# ---------------------------------------------------------------------------

class TestResilientLoading:
    def test_skips_file_without_front_matter(self, tmp_path: Path):
        (tmp_path / "bare.txt").write_text("No front-matter here.", encoding="utf-8")
        write_prompt(tmp_path, "good.txt", name="ner", version="1.0.0", body="ok")
        registry = PromptRegistry(tmp_path)
        assert registry.list_names() == ["ner"]

    def test_skips_file_with_invalid_yaml(self, tmp_path: Path):
        bad = "---\nname: x\nversion: {broken: [yaml\n---\nbody"
        (tmp_path / "bad.txt").write_text(bad, encoding="utf-8")
        write_prompt(tmp_path, "good.txt", name="ner", version="1.0.0", body="ok")
        registry = PromptRegistry(tmp_path)
        assert "x" not in registry.list_names()

    def test_skips_file_missing_name_field(self, tmp_path: Path):
        content = "---\nversion: 1.0.0\n---\nbody"
        (tmp_path / "no-name.txt").write_text(content, encoding="utf-8")
        registry = PromptRegistry(tmp_path)
        assert registry.list_names() == []

    def test_skips_file_missing_version_field(self, tmp_path: Path):
        content = "---\nname: ner\n---\nbody"
        (tmp_path / "no-version.txt").write_text(content, encoding="utf-8")
        registry = PromptRegistry(tmp_path)
        assert registry.list_names() == []

    def test_body_leading_newline_stripped(self, tmp_path: Path):
        write_prompt(tmp_path, "p.txt", name="ner", version="1.0.0", body="hello")
        registry = PromptRegistry(tmp_path)
        assert not registry.get("ner").startswith("\n")

    def test_non_txt_files_ignored(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text("---\nname: x\nversion: 1.0.0\n---\nbody")
        registry = PromptRegistry(tmp_path)
        assert registry.list_names() == []
