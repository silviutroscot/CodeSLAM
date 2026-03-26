import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:  # pragma: no branch
    from codeslam.config import LossConfig
    from codeslam.optimization import StateLayout, build_prior_factor
    from codeslam.types import PriorFactor


@unittest.skipUnless(torch is not None, "torch is required")
class PriorTest(unittest.TestCase):
    def test_prior_shapes(self) -> None:
        prior = PriorFactor(
            keyframe_ids=("a", "b"),
            pose_reference={"a": torch.eye(4), "b": torch.eye(4)},
            code_reference={"a": torch.zeros(128), "b": torch.zeros(128)},
            sqrt_hessian=torch.eye(268),
            offset=torch.zeros(268),
        )
        self.assertEqual(prior.sqrt_hessian.shape, (268, 268))
        self.assertEqual(prior.offset.shape, (268,))

    def test_build_prior_factor_rejects_empty_marginalization(self) -> None:
        layout = StateLayout(("a", "b"), code_dim=2)

        def residual_fn(delta: torch.Tensor) -> torch.Tensor:
            return delta

        with self.assertRaises(ValueError):
            build_prior_factor(
                residual_fn,
                base_poses={"a": torch.eye(4), "b": torch.eye(4)},
                base_codes={"a": torch.zeros(2), "b": torch.zeros(2)},
                layout=layout,
                marginalize_keyframe_ids=set(),
                loss_config=LossConfig(),
            )


if __name__ == "__main__":
    unittest.main()
