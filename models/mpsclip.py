import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class AllGather(torch.autograd.Function):
    """Gather features across workers while preserving local gradients."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.batch_size * ctx.rank
        return grad_output[start:start + ctx.batch_size], None, None


def gather_features(tensor):
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    return AllGather.apply(tensor, dist.get_rank(), dist.get_world_size())


class MPSCLIPBase(nn.Module):
    """Shared MPS-CLIP objectives used by the retrieval model."""

    def __init__(self, config):
        super().__init__()
        if config is None or not config.get('is_mps', False):
            raise ValueError("MPSCLIP requires config['is_mps'] = True")
        self.embed_dim = int(config['embed_dim'])
        self.register_buffer('temp', torch.tensor(float(config['temp1'])))

    def get_contr_loss(self, image_feat, text_feat, idx=None):
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = gather_features(image_feat)
        text_feat_all = gather_features(text_feat)
        logits = image_feat_all @ text_feat_all.t() / self.temp.clamp(min=1e-3)
        batch_size = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(batch_size, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
        else:
            idx_all = gather_features(idx.view(-1, 1))
            positives = torch.eq(idx_all, idx_all.t()).float()
            labels = positives / positives.sum(dim=1, keepdim=True)
            loss_i2t = -(F.log_softmax(logits, dim=1) * labels).sum(dim=1).mean()
            loss_t2i = -(F.log_softmax(logits.t(), dim=1) * labels).sum(dim=1).mean()

        return 0.5 * (loss_i2t + loss_t2i)

    def weighted_triplet_loss(self, image_feat, text_feat, margin=0.2,
                              gamma=2.0, max_violation=False):
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = gather_features(image_feat)
        text_feat_all = gather_features(text_feat)
        scores = image_feat_all @ text_feat_all.t()
        return _weighted_triplet_from_scores(scores, margin, gamma, max_violation)


def multi_view_clip_loss(vision_views, text_feat, logit_scale):
    if not vision_views:
        raise ValueError('vision_views must not be empty')

    text_feat_all = gather_features(text_feat)
    vision_views_all = [gather_features(view) for view in vision_views]
    logits_per_view = [
        logit_scale * (view @ text_feat_all.t()) for view in vision_views_all
    ]
    logits = torch.stack(logits_per_view, dim=0).max(dim=0).values
    labels = torch.arange(text_feat_all.size(0), device=text_feat.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i), logits


def multi_view_weighted_triplet_loss(vision_views, text_feat, embed_dim,
                                     margin=0.2, gamma=2.0,
                                     max_violation=False, reduce_views='max'):
    if not vision_views:
        raise ValueError('vision_views must not be empty')
    if text_feat.size(1) != embed_dim:
        raise ValueError('text feature dimension does not match embed_dim')

    text_feat_all = gather_features(text_feat)
    score_views = []
    for view in vision_views:
        if view.size(1) != embed_dim:
            raise ValueError('vision feature dimension does not match embed_dim')
        score_views.append(gather_features(view) @ text_feat_all.t())

    scores = torch.stack(score_views, dim=0)
    if reduce_views == 'max':
        scores = scores.max(dim=0).values
    elif reduce_views == 'mean':
        scores = scores.mean(dim=0)
    else:
        raise ValueError(f'Unknown reduce_views mode: {reduce_views}')

    return _weighted_triplet_from_scores(scores, margin, gamma, max_violation)


def _weighted_triplet_from_scores(scores, margin, gamma, max_violation):
    diagonal = scores.diag().view(-1, 1)
    cost_text = (margin + scores - diagonal).clamp(min=0)
    cost_image = (margin + scores - diagonal.t()).clamp(min=0)

    diagonal_mask = torch.eye(scores.size(0), device=scores.device, dtype=torch.bool)
    cost_text = cost_text.masked_fill(diagonal_mask, 0)
    cost_image = cost_image.masked_fill(diagonal_mask, 0)

    cost_text = (1 - torch.exp(-cost_text)).pow(gamma) * cost_text
    cost_image = (1 - torch.exp(-cost_image)).pow(gamma) * cost_image

    if max_violation:
        cost_text = cost_text.max(dim=1).values
        cost_image = cost_image.max(dim=0).values

    return 0.5 * (cost_text.sum() + cost_image.sum())
