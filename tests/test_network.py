import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:  # pragma: no branch
    from codeslam.config import ModelConfig
    from codeslam.network import CodeSLAMDepthModel


@unittest.skipUnless(torch is not None, "torch is required")
class NetworkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            input_height=64,
            input_width=96,
            code_dim=8,
            base_channels=4,
            pyramid_levels=4,
            latent_hidden_dim=16,
        )
        self.model = CodeSLAMDepthModel(self.config).eval()

    def test_forward_shapes(self) -> None:
        intensity = torch.rand(2, 1, self.config.input_height, self.config.input_width)
        depth = torch.rand(2, 1, self.config.input_height, self.config.input_width) * 5.0 + 0.5
        prediction = self.model(intensity, depth, sample_posterior=False)

        self.assertEqual(len(prediction.depth_pyramid), self.config.pyramid_levels)
        self.assertEqual(prediction.code.shape, (2, self.config.code_dim))
        self.assertEqual(prediction.posterior_mean.shape, (2, self.config.code_dim))
        self.assertEqual(prediction.posterior_logvar.shape, (2, self.config.code_dim))
        self.assertEqual(prediction.depth_pyramid[-1].shape[-2:], (self.config.input_height // 2, self.config.input_width // 2))
        self.assertTrue(torch.all(prediction.scale_pyramid[-1] > 0))

    def test_linear_jacobian_shape(self) -> None:
        intensity = torch.rand(1, 1, self.config.input_height, self.config.input_width)
        jacobian = self.model.precompute_linear_jacobian(intensity, level=-1, chunk_size=4)
        expected_rows = (self.config.input_height // 2) * (self.config.input_width // 2)
        self.assertEqual(jacobian.shape, (expected_rows, self.config.code_dim))


if __name__ == "__main__":
    unittest.main()
