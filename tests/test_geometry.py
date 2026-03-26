import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:  # pragma: no branch
    from codeslam.config import CameraModel
    from codeslam.geometry import backproject, inside_image_mask, project


@unittest.skipUnless(torch is not None, "torch is required")
class GeometryTest(unittest.TestCase):
    def test_backproject_project_round_trip(self) -> None:
        camera = CameraModel(fx=100.0, fy=120.0, cx=1.0, cy=0.5, width=4, height=3)
        depth = torch.tensor([[[[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]]])
        points = backproject(depth, camera)
        uv, z = project(points, camera)

        expected_x = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
        expected_y = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
        self.assertTrue(torch.allclose(uv[0, ..., 0], expected_x, atol=1e-5))
        self.assertTrue(torch.allclose(uv[0, ..., 1], expected_y, atol=1e-5))
        self.assertTrue(torch.allclose(z[0], depth[0, 0], atol=1e-5))

    def test_inside_image_mask(self) -> None:
        camera = CameraModel(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=4, height=3)
        uv = torch.tensor([[[[0.0, 0.0], [3.0, 2.0], [4.0, 2.0], [-1.0, 0.0]]]])
        mask = inside_image_mask(uv, camera)
        self.assertEqual(mask.tolist(), [[[True, True, False, False]]])


if __name__ == "__main__":
    unittest.main()
