from __future__ import annotations

import unittest

import preprocessing


class PreprocessingTest(unittest.TestCase):
    def test_scale_depth_updates_boundary_bins(self) -> None:
        histogram = [10.0, 20.0, 30.0]
        preprocessing.scale_depth(histogram, average=5.0)
        self.assertAlmostEqual(histogram[0], 5.0 / 15.0)
        self.assertEqual(histogram[1], 20.0)
        self.assertAlmostEqual(histogram[2], 5.0 / 35.0)

    def test_parse_args_supports_multiple_actions(self) -> None:
        args = preprocessing.parse_args(["i", "r", "--width", "320", "--height", "240", "--data-root", "dataset"])
        self.assertEqual(args.actions, ["i", "r"])
        self.assertEqual(args.width, 320)
        self.assertEqual(args.height, 240)
        self.assertEqual(args.data_root, "dataset")


if __name__ == "__main__":
    unittest.main()
