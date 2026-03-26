from __future__ import annotations

import unittest

from codeslam.config import ModelConfig, SLAMConfig, TrainingConfig


class ConfigTest(unittest.TestCase):
    def test_proximity_alias_matches_average_depth(self) -> None:
        config = ModelConfig(proximity_average_depth=3.5)
        self.assertEqual(config.proximity_transition, 3.5)

    def test_training_defaults_match_paper_schedule(self) -> None:
        config = TrainingConfig()
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.final_learning_rate, 1e-6)
        self.assertEqual(config.epochs, 6)

    def test_slam_defaults_match_paper_setup(self) -> None:
        config = SLAMConfig()
        self.assertEqual(config.max_keyframes, 4)
        self.assertEqual(config.mapping_edges, "consecutive")
        self.assertEqual(config.bootstrap_motion_prior, 0.0)


if __name__ == "__main__":
    unittest.main()
