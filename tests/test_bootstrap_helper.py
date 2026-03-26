from __future__ import annotations

import sys
import unittest
from pathlib import Path

from scripts._bootstrap import ensure_repo_root_on_path


class BootstrapHelperTest(unittest.TestCase):
    def test_repo_root_is_added_to_sys_path(self) -> None:
        root = ensure_repo_root_on_path()
        self.assertIsInstance(root, Path)
        self.assertEqual(root, Path(__file__).resolve().parents[1])
        self.assertIn(str(root), sys.path)


if __name__ == "__main__":
    unittest.main()
