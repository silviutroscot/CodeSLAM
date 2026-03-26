from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_compute_errors_module():
    module_path = Path(__file__).resolve().parents[1] / "U-Net" / "compute_errors.py"
    spec = importlib.util.spec_from_file_location("legacy_compute_errors", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ComputeErrorsTest(unittest.TestCase):
    def test_collect_test_files_sorts_images_and_labels(self) -> None:
        module = load_compute_errors_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for folder in ["1", "0"]:
                (root / folder / "photo").mkdir(parents=True, exist_ok=True)
                (root / folder / "depth").mkdir(parents=True, exist_ok=True)
            for relative in [
                ("1", "photo", "b.jpg"),
                ("0", "photo", "a.jpg"),
                ("1", "depth", "b.png"),
                ("0", "depth", "a.png"),
            ]:
                (root / relative[0] / relative[1] / relative[2]).touch()

            images, labels = module.collect_test_files(str(root))
            self.assertEqual([Path(path).name for path in images], ["a.jpg", "b.jpg"])
            self.assertEqual([Path(path).name for path in labels], ["a.png", "b.png"])

    def test_parse_args_accepts_override(self) -> None:
        module = load_compute_errors_module()
        args = module.parse_args(["--test-dir", "dataset"])
        self.assertEqual(args.test_dir, "dataset")


if __name__ == "__main__":
    unittest.main()
