import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import open_clip
from open_clip import tokenizer
import nltk
from nltk import pos_tag, word_tokenize

from models import MPSCLIP, load_pretrained_model
from .MVR import  MVR
from .mpsclip import multi_view_clip_loss, multi_view_weighted_triplet_loss


class MPS_CLIP(MPSCLIP):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True,
                         use_contrastive_loss=True, use_affil_loss=False)
        self.config = config
        self.use_affil_loss = config['use_affil_loss']
        self.use_triplet_loss = config['use_triplet_loss']
        self.create_and_load_pretrained(config)
        self.align_before = False
        self.use_multi_view = config['use_multi_view']
        self.mv_k = config['mv_k']
        self.mv_use_mlp = config['mv_mlp']
        self.mv_dropout = config['mv_dropout']
        


        
        self.mv_margin = float(config['mv_margin'])
        self.mv_weight = float(config['mv_weight'])  
        self.s_base_contr = nn.Parameter(torch.zeros(1))  
        self.s_mv_contr   = nn.Parameter(torch.zeros(1))
        self.s_base_trip = nn.Parameter(torch.zeros(1))
        self.s_mv_trip   = nn.Parameter(torch.zeros(1))
        
       
      
        if self.use_multi_view:
            embed_dim = self._infer_embed_dim()
            self.mv_img_head =MVR(
                in_dim=embed_dim, out_dim=embed_dim, k=self.mv_k,
                mlp=self.mv_use_mlp, dropout=self.mv_dropout
            )
        else:
            self.mv_img_head = None
        
        
       
    def _uncertainty_combine(self, L_base, L_mv, s_base, s_mv):
        sigma2_base = torch.exp(s_base)
        sigma2_mv   = torch.exp(s_mv)

        loss = (L_base / (2.0 * sigma2_base) +
                L_mv   / (2.0 * sigma2_mv)   +
                0.5 * (s_base + s_mv))      
        return loss


    def create_and_load_pretrained(self, config):
        if self.config['model'] == 'geo':
            self.model, _, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained='openai')
            if not self.config['if_evaluation']:
                ckpt_path = "/gpfs-flash/hulab/likai_srt/lyf/code/HarMA/models/pretrain/RS5M_ViT-B-32_RET-2.pt"
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                msg = self.model.load_state_dict(checkpoint, strict=False)
        else:
            self.model, _, _ = open_clip.create_model_and_transforms("ViT-B/32")

    def get_vis_emb(self, image, seperate=False):
        if self.config['is_harma']:
            if self.align_before:
                img_emb, feas_vis = self.model.encode_image(image, normalize=True)
                return img_emb, feas_vis
            else:
                if seperate:
                    features = self.model.encode_image1(image, normalize=True)
                    if isinstance(features, tuple):
                        patch_feats, global_feat = features
                        return patch_feats, global_feat
                    else:
                        return None, features
                else:
                    img_emb = self.model.encode_image(image, normalize=True)
                    return img_emb

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
        if self.config['is_harma']:
            if self.align_before:
                txt_emb, feas_txt = self.model.encode_text(text_ids, normalize=True)
                return txt_emb, feas_txt
            else:
                txt_emb = self.model.encode_text(text_ids, normalize=True)
            return txt_emb

    def get_txt_emb1(self, text_ids, return_word_feats=False):
        if return_word_feats:
            word_feats, sent_feat = self.model.encode_text1(text_ids, normalize=True, return_word_feats=True)
            return word_feats, sent_feat
        else:
            sent_feat = self.model.encode_text(text_ids, normalize=True)
            return None, sent_feat

    def extract_nouns(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        nouns = [word for word, tag in tagged if tag.startswith('NN')]
        return nouns if nouns else [tokens[0]]

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

        
        img_base = global_img_feat   
        
        base_contr, base_triplet, base_affil = None, None, None
        if self.use_affil_loss:
            base_contr = self.get_contr_loss(img_base, global_txt_feat,
                                             idx=idx, label=label, config=self.config)
            base_affil = self.get_affil_loss(img_base, global_txt_feat,
                                             idx=idx, label=label, config=self.config)
        elif self.use_triplet_loss:
            base_triplet = self.get_triplet_loss(img_base, global_txt_feat)
        else:
            base_contr = self.get_contr_loss(img_base, global_txt_feat, idx)
            base_triplet = self.weighted_triplet_loss(img_base, global_txt_feat)


        mv_contr, mv_triplet = None, None
        if self.use_multi_view and (region_global_feat is not None):
            v_list = self.mv_img_head(region_global_feat)
            logit_scale = self._get_logit_scale(global_txt_feat.device)
            mv_contr, _ = multi_view_clip_loss(v_list, global_txt_feat, logit_scale)
            mv_triplet = multi_view_weighted_triplet_loss(self, v_list, global_txt_feat, margin=self.mv_margin, gamma=2.0, max_violation=False, reduce_views="max")

        if self.use_affil_loss:
            if self.use_multi_view and (mv_contr is not None):
                total_contr = (1 - self.mv_contr_weight) * base_contr + self.mv_contr_weight * mv_contr
            else:
                total_contr = base_contr
            return total_contr, base_affil

        elif self.use_triplet_loss:
            if self.use_multi_view and (mv_triplet is not None):
                total_triplet = (1 - self.mv_triplet_weight) * base_triplet + self.mv_triplet_weight * mv_triplet
            else:
                total_triplet = base_triplet
            return total_triplet

        else:
            if self.use_multi_view and (mv_contr is not None) and (mv_triplet is not None):
                total_contr = self._uncertainty_combine(
                    base_contr, mv_contr, self.s_base_contr, self.s_mv_contr
                )
                # total_contr = base_contr
                total_triplet = self._uncertainty_combine(
                    base_triplet, mv_triplet, self.s_base_trip, self.s_mv_trip
                )
            else:
                total_contr, total_triplet = base_contr, base_triplet

            return total_contr, total_triplet

