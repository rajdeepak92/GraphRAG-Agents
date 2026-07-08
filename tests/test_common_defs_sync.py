from __future__ import annotations

import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


class CommonDefsSyncTests(unittest.TestCase):
    """The runtime imports the repo-root ``common_defs``; mypy resolves ``src/common_defs``.

    The two files must stay byte-identical so that any shared literal (e.g. an ``EnvVar``
    member) exists in both the runtime and the type-checked copy. This guard fails loudly
    if a future edit updates only one of them.
    """

    def test_root_and_src_common_defs_are_identical(self) -> None:
        root = _REPO_ROOT / "common_defs.py"
        src = _REPO_ROOT / "src" / "common_defs.py"
        self.assertEqual(
            root.read_text(encoding="utf-8"),
            src.read_text(encoding="utf-8"),
            "common_defs.py and src/common_defs.py have drifted; keep them in lockstep.",
        )


if __name__ == "__main__":
    unittest.main()
