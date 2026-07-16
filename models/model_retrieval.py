import os

import torch
import torch.nn.functional as F
import open_clip

from .mpr import MultiPerspectiveRepresentation
from .mpsclip import MPSCLIPBase, multi_view_clip_loss, multi_view_weighted_triplet_loss


class MPSCLIP(MPSCLIPBase):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.create_and_load_pretrained(config)
        self.align_before = False
        self.use_multi_view = config.get('use_multi_view', True)
        self.mv_k = config.get('mv_k', 4)
        self.mv_use_mlp = config.get('mv_mlp', True)
        self.mv_dropout = config.get('mv_dropout', 0.0)
        self.mv_margin = float(config.get('mv_margin', 0.2))
        self.lambda_mpc = float(config.get('lambda_mpc', 0.2))
        self.lambda_mpt = float(config.get('lambda_mpt', 0.2))

        if self.use_multi_view:
            embed_dim = self._infer_embed_dim()
            self.mpr_head = MultiPerspectiveRepresentation(
                in_dim=embed_dim, out_dim=embed_dim, k=self.mv_k,
                mlp=self.mv_use_mlp, dropout=self.mv_dropout
            )
        else:
            self.mpr_head = None

    def create_and_load_pretrained(self, config):
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B/32", pretrained='openai'
        )

        if config['model'] == 'geo':
            ckpt_path = config.get(
                'pretrained_path',
                'models/pretrain/RS5M_ViT-B-32_RET-2.pt',
            )
            if os.path.isfile(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint.get('model', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
            elif not config.get('if_evaluation', False):
                raise FileNotFoundError(
                    f"GeoRSCLIP checkpoint not found: {ckpt_path}. "
                    "Download it as described in README.md or set pretrained_path."
                )

    def get_vis_emb(self, image, seperate=False):
        if self.align_before:
            img_emb, feas_vis = self.model.encode_image(image, normalize=True)
            return img_emb, feas_vis
        if seperate:
            features = self.model.encode_image1(image, normalize=True)
            if isinstance(features, tuple):
                patch_feats, global_feat = features
                return patch_feats, global_feat
            else:
                return None, features
        return self.model.encode_image(image, normalize=True)

    def get_vis_emb1(self, image):
        features = self.model.encode_image1(image, normalize=True)
        if isinstance(features, tuple):
            patch_feats, global_feat = features
            return patch_feats, global_feat
        else:
            return None, features

    def get_region_vis_emb(self, regions, region_masks):
        """
        regions:      [B, max_n, C, H, W]
        region_masks: [B, max_n]  bool
        返回: region_global_feat: [B, D]
        """
        B, max_n, C, H, W = regions.shape
        if max_n == 0:
            return None

        regions_flat = regions.view(B * max_n, C, H, W)
        _, global_region_flat = self.get_vis_emb(regions_flat, seperate=True)  # [B*max_n, D]

        D = global_region_flat.shape[-1]
        global_region = global_region_flat.view(B, max_n, D)  # [B,max_n,D]

        mask = region_masks.unsqueeze(-1).float()             # [B,max_n,1]
        global_region = global_region * mask                  # padding 置 0

        denom = mask.sum(dim=1).clamp(min=1.0)                # [B,1]
        region_global_feat = global_region.sum(dim=1) / denom # [B,D]

        region_global_feat = F.normalize(region_global_feat, dim=-1)
        return region_global_feat

    def get_txt_emb(self, text_ids, idx=None, label=None):
        if self.align_before:
            txt_emb, feas_txt = self.model.encode_text(text_ids, normalize=True)
            return txt_emb, feas_txt
        return self.model.encode_text(text_ids, normalize=True)

    def get_txt_emb1(self, text_ids, return_word_feats=False):
        if return_word_feats:
            word_feats, sent_feat = self.model.encode_text1(text_ids, normalize=True, return_word_feats=True)
            return word_feats, sent_feat
        else:
            sent_feat = self.model.encode_text(text_ids, normalize=True)
            return None, sent_feat

    def _infer_embed_dim(self):
        if hasattr(self.model, "embed_dim"):
            return int(self.model.embed_dim)
        if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
            return int(self.model.text_projection.shape[-1])
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "output_dim"):
            return int(self.model.visual.output_dim)
        if 'embed_dim' in self.config:
            return int(self.config['embed_dim'])
        raise RuntimeError("Cannot infer CLIP embed dim. Please set config['embed_dim'].")

    def _get_logit_scale(self, device):
        if hasattr(self.model, "logit_scale"):
            return self.model.logit_scale.exp()
        return torch.tensor(1.0, device=device)


    def forward(self, image, regions, region_masks, text_ids, raw_texts,
                idx=None, label=None, num_regions=None):


        _, global_img_feat = self.get_vis_emb(image, seperate=True)          # [B,D]
        _, global_txt_feat = self.get_txt_emb1(text_ids, return_word_feats=True)  # [B,L,D], [B,D]



        region_global_feat = None
        if (regions is not None) and (regions.numel() > 0):
            region_global_feat = self.get_region_vis_emb(regions, region_masks)  # [B,D]
            has_regions = region_masks.any(dim=1, keepdim=True)
            region_global_feat = torch.where(
                has_regions, region_global_feat, global_img_feat
            )


        img_base = global_img_feat

        base_contr = self.get_contr_loss(img_base, global_txt_feat, idx)
        base_triplet = self.weighted_triplet_loss(img_base, global_txt_feat)


        mv_contr, mv_triplet = None, None
        if self.use_multi_view and (region_global_feat is not None):
            v_list = self.mpr_head(region_global_feat)
            logit_scale = self._get_logit_scale(global_txt_feat.device)
            mv_contr, _ = multi_view_clip_loss(v_list, global_txt_feat, logit_scale)
            mv_triplet = multi_view_weighted_triplet_loss(
                v_list,
                global_txt_feat,
                embed_dim=self.embed_dim,
                margin=self.mv_margin,
                gamma=2.0,
                max_violation=False,
                reduce_views="max",
            )

        if self.use_multi_view and (mv_contr is not None) and (mv_triplet is not None):
            total_contr = base_contr + self.lambda_mpc * mv_contr
            total_triplet = base_triplet + self.lambda_mpt * mv_triplet
        else:
            total_contr, total_triplet = base_contr, base_triplet

        return total_contr, total_triplet
