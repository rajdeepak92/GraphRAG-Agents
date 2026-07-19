"""P2 tests: no-Git direct file transaction and rollback journal (plan §6).

Covers created-file deletion, exact shared-file restoration, crash-resume
reconciliation, the write allowlist, and proof that zero ``git`` executables are
launched during a transaction.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from multi_agentic_graph_rag.services.direct_file_transaction import (
    DirectFileTransaction,
    FileDriftError,
    FileTransactionError,
    JournalStatus,
    PathPolicyError,
    classify_path,
    sha256_bytes,
)

TEST_FILE = "tests/sensor/Tc100001ValidateTemperatureSensorThreshold.py"
WRAPPER = "test_lib/sensor/sensor_wrappers.py"


def _framework(tmp_path: Path) -> Path:
    root = tmp_path / "framework"
    (root / "test_lib" / "sensor").mkdir(parents=True)
    return root


def _tx(root: Path, tmp_path: Path, tc_id: int = 100001) -> DirectFileTransaction:
    return DirectFileTransaction(
        framework_root=root,
        project="demo",
        run_id="RUN-1",
        tc_id=tc_id,
        journal_root=tmp_path / "journals",
    )


# --- allowlist ---------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "/etc/passwd",
        "../secrets.py",
        "tests/sensor/../../escape.py",
        "src/production/module.py",
        "test_lib/sensor/utils.py",  # generic dumping ground forbidden
        "test_lib/sensor/network_wrappers.py",  # module prefix must match dir
        "tests/sensor/helper.py",  # not a Tc<6-digit> stem
        r"tests\sensor\Tc100001X.py",  # backslash
    ],
)
def test_classify_rejects_disallowed_paths(bad: str) -> None:
    with pytest.raises(PathPolicyError):
        classify_path(bad)


def test_classify_accepts_permitted_shapes() -> None:
    assert classify_path(TEST_FILE).role == "test"
    assert classify_path(WRAPPER).role == "wrapper"
    assert classify_path("test_lib/sensor/sensor_helpers.py").role == "helper"
    assert classify_path("tests_robot/sensor/Tc100001X.robot").role == "robot"


@pytest.mark.parametrize(
    "path",
    [
        "tests/sensor/Tc000001X.py",
        "tests/sensor/Tc100000X.py",
        "tests/sensor/Tc100001lowercase.py",
    ],
)
def test_classify_rejects_out_of_range_or_non_pascal_tc_stems(path: str) -> None:
    with pytest.raises(PathPolicyError):
        classify_path(path)


def test_transaction_rejects_unsafe_journal_components_and_tc_ids(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    with pytest.raises(PathPolicyError):
        DirectFileTransaction(
            framework_root=root,
            project="../demo",
            run_id="RUN-1",
            tc_id=100001,
            journal_root=tmp_path / "journals",
        )
    with pytest.raises(PathPolicyError):
        DirectFileTransaction(
            framework_root=root,
            project="demo",
            run_id="RUN/1",
            tc_id=100001,
            journal_root=tmp_path / "journals",
        )
    with pytest.raises(PathPolicyError):
        _tx(root, tmp_path, tc_id=100000)


def test_path_tc_must_match_transaction_tc(tmp_path: Path) -> None:
    tx = _tx(_framework(tmp_path), tmp_path, tc_id=100002)
    tx.begin()
    with pytest.raises(PathPolicyError, match="does not match"):
        tx.write(TEST_FILE, "pass\n")


def test_transaction_rejects_internal_symlink_indirection(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    production = root / "production"
    production.mkdir()
    (root / "tests").mkdir()
    try:
        (root / "tests" / "sensor").symlink_to(production, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    tx = _tx(root, tmp_path)
    tx.begin()
    with pytest.raises(PathPolicyError, match="symlink"):
        tx.write(TEST_FILE, "pass\n")
    assert not (production / Path(TEST_FILE).name).exists()


# --- rollback ----------------------------------------------------------------


def test_rollback_deletes_created_files(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('generated')\n")
    assert (root / TEST_FILE).exists()

    tx.rollback()
    assert not (root / TEST_FILE).exists()
    # The module directory the transaction created is pruned when left empty.
    assert not (root / "tests" / "sensor").exists()
    assert tx.journal.status is JournalStatus.ROLLED_BACK


def test_write_requires_begin(tmp_path: Path) -> None:
    tx = _tx(_framework(tmp_path), tmp_path)
    with pytest.raises(FileTransactionError, match="must begin"):
        tx.write(TEST_FILE, "pass\n")


def test_rollback_restores_exact_shared_preimage(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    original = "def existing_wrapper():\n    return True\n"
    (root / WRAPPER).write_text(original, encoding="utf-8")
    original_bytes = (root / WRAPPER).read_bytes()
    original_hash = sha256_bytes(original_bytes)

    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('t')\n")
    tx.write(WRAPPER, original + "\n\ndef new_wrapper():\n    return False\n")
    assert (root / WRAPPER).read_text(encoding="utf-8") != original

    tx.rollback()
    # Shared file restored byte-for-byte; created test file removed.
    assert (root / WRAPPER).read_text(encoding="utf-8") == original
    assert sha256_bytes((root / WRAPPER).read_bytes()) == original_hash
    assert not (root / TEST_FILE).exists()


def test_seal_keeps_files_and_blocks_rollback(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('kept')\n")
    tx.seal()
    assert (root / TEST_FILE).exists()
    assert tx.journal.status is JournalStatus.SEALED


def test_create_only_test_file_cannot_overwrite_accepted(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    (root / "tests" / "sensor").mkdir(parents=True)
    (root / TEST_FILE).write_text("print('accepted')\n", encoding="utf-8")
    tx = _tx(root, tmp_path)
    tx.begin()
    with pytest.raises(PathPolicyError, match="accepted TC"):
        tx.write(TEST_FILE, "print('overwrite')\n")


def test_existing_package_initializer_is_recorded_without_rewrite(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    initializer = root / "test_lib" / "sensor" / "__init__.py"
    initializer.write_text("\n", encoding="utf-8")
    tx = _tx(root, tmp_path)
    tx.begin()

    tx.write("test_lib/sensor/__init__.py", "\n")

    assert initializer.read_text(encoding="utf-8") == "\n"
    assert tx.journal.expected_file_hashes()["test_lib/sensor/__init__.py"] == sha256_bytes(
        initializer.read_bytes()
    )


def test_existing_package_initializer_ignores_requested_changes(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    initializer = root / "test_lib" / "sensor" / "__init__.py"
    initializer.write_text("\n", encoding="utf-8")
    tx = _tx(root, tmp_path)
    tx.begin()

    tx.write("test_lib/sensor/__init__.py", "from .sensor_helpers import helper\n")

    assert initializer.read_text(encoding="utf-8") == "\n"


def test_crash_resume_reconciles_open_journal(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('partial')\n")
    # Simulate a crash: a fresh transaction object over the same journal dir.
    resumed = _tx(root, tmp_path)
    status = resumed.reconcile()
    assert status is JournalStatus.ROLLED_BACK
    assert not (root / TEST_FILE).exists()


def test_intent_is_durable_before_framework_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    real_write = tx._atomic_write

    def _crash_after_replace(target: Path, data: bytes) -> None:
        # The journal entry and created-directory intent must already be durable.
        state = json.loads(tx.journal._state_path.read_text(encoding="utf-8"))
        assert state["entries"][0]["relative_path"] == TEST_FILE
        assert "tests/sensor" in state["created_dirs"]
        real_write(target, data)
        raise RuntimeError("simulated process termination")

    monkeypatch.setattr(tx, "_atomic_write", _crash_after_replace)
    with pytest.raises(RuntimeError, match="termination"):
        tx.write(TEST_FILE, "print('partial')\n")

    resumed = _tx(root, tmp_path)
    assert resumed.reconcile() is JournalStatus.ROLLED_BACK
    assert not (root / TEST_FILE).exists()


def test_same_write_replay_is_idempotent(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('same')\n")
    tx.write(TEST_FILE, "print('same')\n")
    assert len(tx.journal.entries) == 1


def test_repair_can_replace_only_a_case_owned_create(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('first')\n")
    tx.write(TEST_FILE, "print('repaired')\n")
    assert (root / TEST_FILE).read_text(encoding="utf-8") == "print('repaired')\n"
    assert len(tx.journal.entries) == 2
    tx.rollback()
    assert not (root / TEST_FILE).exists()


def test_repair_refuses_to_overwrite_external_drift(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('generated')\n")
    (root / TEST_FILE).write_text("print('someone else')\n", encoding="utf-8")

    with pytest.raises(FileDriftError, match="refusing to overwrite drifted"):
        tx.write(TEST_FILE, "print('repaired')\n")

    assert (root / TEST_FILE).read_text(encoding="utf-8") == "print('someone else')\n"
    assert len(tx.journal.entries) == 1


def test_rollback_refuses_to_clobber_drifted_created_file(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('generated')\n")
    (root / TEST_FILE).write_text("print('someone else')\n", encoding="utf-8")
    with pytest.raises(FileDriftError, match="drifted"):
        tx.rollback()
    assert (root / TEST_FILE).read_text(encoding="utf-8") == "print('someone else')\n"
    assert tx.journal.status is JournalStatus.OPEN


def test_reconcile_revalidates_tampered_journal_paths(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('generated')\n")
    state = json.loads(tx.journal._state_path.read_text(encoding="utf-8"))
    state["entries"][0]["relative_path"] = "../victim.py"
    tx.journal._state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(PathPolicyError):
        _tx(root, tmp_path).reconcile()
    assert (root / TEST_FILE).exists()


def test_reconcile_can_seal_an_externally_finalized_case(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('accepted')\n")
    resumed = _tx(root, tmp_path)
    calls: list[int] = []

    def _finalized(journal) -> bool:  # type: ignore[no-untyped-def]
        calls.append(journal.tc_id)
        return True

    assert resumed.reconcile(finalization_check=_finalized) is JournalStatus.SEALED
    assert calls == [100001]
    assert (root / TEST_FILE).exists()


def test_journal_exposes_original_shared_preimages_and_final_hashes(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    original = "def existing_wrapper():\n    return True\n"
    (root / WRAPPER).write_text(original, encoding="utf-8")
    original_bytes = (root / WRAPPER).read_bytes()
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(WRAPPER, original + "\n\ndef added():\n    return True\n")
    tx.write(WRAPPER, original + "\n\ndef repaired():\n    return True\n")
    assert tx.journal.validation_preimages() == {WRAPPER: original_bytes}
    assert tx.journal.expected_file_hashes()[WRAPPER] == sha256_bytes((root / WRAPPER).read_bytes())
    assert tx.journal.metadata()["entry_count"] == 2


def test_reconcile_leaves_sealed_journal_untouched(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('accepted')\n")
    tx.seal()
    resumed = _tx(root, tmp_path)
    assert resumed.reconcile() is JournalStatus.SEALED
    assert (root / TEST_FILE).exists()


def test_rolled_back_journal_can_restart_with_same_permanent_tc_id(tmp_path: Path) -> None:
    root = _framework(tmp_path)
    first = _tx(root, tmp_path)
    first.begin()
    first.write(TEST_FILE, "print('first')\n")
    first.rollback()

    resumed = _tx(root, tmp_path)
    resumed.begin()
    resumed.restart()
    resumed.write(TEST_FILE, "print('resumed')\n")
    assert resumed.journal.attempt == 2
    assert resumed.journal.status is JournalStatus.OPEN
    assert (resumed.journal.journal_dir / "journal.attempt-0001.json").is_file()
    resumed.rollback()
    assert not (root / TEST_FILE).exists()


# --- no Git ------------------------------------------------------------------


def test_transaction_launches_no_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real_run = subprocess.run

    def _guard(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        name = cmd[0] if isinstance(cmd, list | tuple) else str(cmd)
        calls.append(str(name))
        if "git" in str(name).lower():
            raise AssertionError(f"git was invoked: {cmd}")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _guard)
    root = _framework(tmp_path)
    tx = _tx(root, tmp_path)
    tx.begin()
    tx.write(TEST_FILE, "print('t')\n")
    tx.write(WRAPPER, "def w():\n    return True\n")
    tx.rollback()
    assert all("git" not in call.lower() for call in calls)


def test_module_source_imports_no_git_library() -> None:
    from multi_agentic_graph_rag.services import direct_file_transaction

    source = Path(direct_file_transaction.__file__).read_text(encoding="utf-8")
    for forbidden in ("import git", "from git", "dulwich", "pygit2", "subprocess"):
        assert forbidden not in source, f"forbidden Git/subprocess reference: {forbidden}"
