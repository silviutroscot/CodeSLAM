import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:  # pragma: no branch
    from codeslam.pose import inverse, se3_exp, se3_log, so3_log, transform_points


@unittest.skipUnless(torch is not None, "torch is required")
class PoseTest(unittest.TestCase):
    def test_exp_log_are_consistent(self) -> None:
        twist = torch.tensor([0.1, -0.2, 0.05, 0.01, -0.02, 0.03], dtype=torch.float32)
        transform = se3_exp(twist)
        recovered = se3_log(transform)
        self.assertTrue(torch.allclose(twist, recovered, atol=1e-4))

    def test_inverse_is_identity(self) -> None:
        twist = torch.tensor([0.1, 0.0, 0.2, 0.03, -0.01, 0.02], dtype=torch.float32)
        transform = se3_exp(twist)
        identity = transform @ inverse(transform)
        self.assertTrue(torch.allclose(identity, torch.eye(4), atol=1e-4))

    def test_so3_log_of_identity_is_zero(self) -> None:
        omega = so3_log(torch.eye(3))
        self.assertTrue(torch.allclose(omega, torch.zeros(3), atol=1e-7))

    def test_transform_points_preserves_batched_points(self) -> None:
        transform = torch.eye(4)
        points = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
        transformed = transform_points(transform, points)
        self.assertEqual(transformed.shape, points.shape)
        self.assertTrue(torch.allclose(transformed, points))


if __name__ == "__main__":
    unittest.main()
