import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:  # pragma: no branch
    from codeslam.proximity import depth_to_proximity, proximity_to_depth


@unittest.skipUnless(torch is not None, "torch is required")
class ProximityTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        depth = torch.tensor([0.5, 2.0, 4.0, 8.0, 20.0], dtype=torch.float32)
        proximity = depth_to_proximity(depth, transition=4.0)
        restored = proximity_to_depth(proximity, transition=4.0)
        self.assertTrue(torch.allclose(depth, restored, atol=1e-5))

    def test_average_depth_maps_to_half_proximity(self) -> None:
        depth = torch.tensor([4.0], dtype=torch.float32)
        proximity = depth_to_proximity(depth, transition=4.0)
        self.assertTrue(torch.allclose(proximity, torch.tensor([0.5]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
