import unittest

import torch
import torch.nn.functional as F

from models.mpr import MultiPerspectiveRepresentation
from models.mpsclip import (
    MPSCLIPBase,
    multi_view_clip_loss,
    multi_view_weighted_triplet_loss,
)


class MPSCLIPLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = MPSCLIPBase({
            'is_mps': True,
            'embed_dim': 8,
            'temp1': 0.07,
        })
        self.image = F.normalize(torch.randn(4, 8, requires_grad=True), dim=-1)
        self.text = F.normalize(torch.randn(4, 8, requires_grad=True), dim=-1)

    def test_single_process_objectives_are_finite_and_differentiable(self):
        contrastive = self.model.get_contr_loss(self.image, self.text)
        triplet = self.model.weighted_triplet_loss(self.image, self.text)
        loss = contrastive + triplet
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.image.grad_fn)

    def test_multi_view_objectives_are_finite_and_differentiable(self):
        views = [self.image, torch.roll(self.image, shifts=1, dims=0)]
        contrastive, logits = multi_view_clip_loss(
            views, self.text, logit_scale=torch.tensor(1.0)
        )
        triplet = multi_view_weighted_triplet_loss(
            views, self.text, embed_dim=8
        )
        loss = contrastive + triplet
        self.assertEqual(tuple(logits.shape), (4, 4))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


class MultiPerspectiveRepresentationTest(unittest.TestCase):
    def test_outputs_have_expected_shape_and_norm(self):
        module = MultiPerspectiveRepresentation(in_dim=8, out_dim=8, k=3)
        outputs = module(torch.randn(4, 8))
        self.assertEqual(len(outputs), 3)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (4, 8))
            torch.testing.assert_close(output.norm(dim=-1), torch.ones(4))


if __name__ == '__main__':
    unittest.main()
