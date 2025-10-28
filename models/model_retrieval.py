import torch
from models import HarMABase, load_pretrained_harma
import torch.nn.functional as F
import torch.nn as nn
import torch
# from torchinfo import summary
from PIL import Image
import open_clip
# from inference_tool import get_preprocess
from open_clip import tokenizer
import nltk
from nltk import pos_tag, word_tokenize
from dinov3.models.vision_transformer import DinoVisionTransformer
# clip, _, _ = open_clip.create_model_and_transforms("ViT-B/32")
# checkpoint = torch.load(ckpt_path, map_location="cpu")
# msg = clip.load_state_dict(checkpoint, strict=False)
# print("Missing keys: ", msg.missing_keys)
# print("Unexpected keys: ", msg.unexpected_keys)
from open_clip.model import MMadapter_img

class HarMA(HarMABase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True, use_contrastive_loss=True, \
                         use_affil_loss=False)
        self.config = config
        self.use_affil_loss = config['use_affil_loss']
        self.use_triplet_loss = config['use_triplet_loss']
        self.create_and_load_pretrained(config)
        self.create_and_load_pretrained_dino(config)
        self.align_before = False
        self.img_proj = nn.Linear(768, self.embed_dim, bias=False)

        

    def create_and_load_pretrained(self, config):
        if self.config['model'] == 'geo': 
            self.model, _ ,_ = open_clip.create_model_and_transforms("ViT-B/32",pretrained='openai')
            if self.config['if_evaluation'] == False:
                ckpt_path = "/gpfs-flash/hulab/likai_srt/lyf/code/HarMA/models/pretrain/RS5M_ViT-B-32_RET-2.pt"
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                msg = self.model.load_state_dict(checkpoint, strict=False)
        else:
            self.model, _, _ = open_clip.create_model_and_transforms("ViT-B/32")


    def create_and_load_pretrained_dino(self, config):
        self.dino = DinoVisionTransformer(
            img_size=224,patch_size=16,
            embed_dim=768,depth=12,
            num_heads=12,ffn_ratio=4.0,qkv_bias=True,drop_path_rate=0.0,
            norm_layer='layernorm',mmadapter=MMadapter_img)
        ckpt_path = self.config['dino_ckpt']
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        msg = self.dino.load_state_dict(checkpoint, strict=False)
        print("missing:", len(msg.missing_keys), "unexpected:", len(msg.unexpected_keys))

           

    def get_vis_emb(self, image, seperate=False):
        if self.config['is_harma']:
            if self.align_before:
                img_emb,feas_vis = self.model.encode_image(image,normalize=True)
                return img_emb,feas_vis
            else:
                # if seperate:
                #     features = self.dino(image)
                #     print(features.shape)
                #     if isinstance(features, tuple):
                #         patch_feats, global_feat = features
                #         return patch_feats, global_feat
                #     else:
                #         return None, features
                if seperate:
                    features = self.dino(image)  # 常见是 [B, 768] 全局特征
                    # 投影到 self.embed_dim，并归一化
                    global_feat = F.normalize(self.img_proj(features), dim=-1)
                    # if isinstance(features, tuple):
                    #     patch_feats, global_feat = features

                    #     if patch_feats is not None:
                    #         patch_feats = F.normalize(self.img_proj(patch_feats), dim=-1)
                        
                    print("global", global_feat.shape)
                        # print("patch", patch_feats.shape)

                    return global_feat
    def get_vis_emb1(self, image):
        features = self.model.encode_image1(image, normalize=True)
        if isinstance(features, tuple):
            patch_feats, global_feat = features
            return patch_feats, global_feat
        else:
            return None, features
        
    def get_txt_emb(self, text_ids, idx=None, label=None):
        if self.config['is_harma']:
            if self.align_before:
                txt_emb,feas_txt = self.model.encode_text(text_ids,normalize=True)
                return txt_emb,feas_txt
            else:
                txt_emb = self.model.encode_text(text_ids,normalize=True)
            return txt_emb
    def get_txt_emb1(self, text_ids, return_word_feats=False):
        if return_word_feats:
            word_feats, sent_feat = self.model.encode_text1(text_ids, normalize=True, return_word_feats=True)
            print("words", word_feats.shape)
            print("sent",  sent_feat.shape)
            return word_feats, sent_feat
        else:
            sent_feat = self.model.encode_text(text_ids, normalize=True, return_word_feats=False)
            return None, sent_feat  
        
    def extract_nouns(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        nouns = [word for word, tag in tagged if tag.startswith('NN')]
        return nouns if nouns else [tokens[0]]


    def forward(self, image, text_ids, raw_texts, idx=None, label=None):
        ## Baseline(Swin-T+Bert-B)
        global_img_feat = self.get_vis_emb(image, seperate=True)
        word_feats, global_txt_feat = self.get_txt_emb1(text_ids, return_word_feats=True)
        noun_feats_list = []  
        for i, text in enumerate(raw_texts):
            text_nouns = self.extract_nouns(text) 
            noun_indices = [j for j, word in enumerate(text.split()) if word in text_nouns]
            if noun_indices:  
                noun_feats_list.append(word_feats[i, noun_indices]) 
            else:
                noun_feats_list.append(word_feats[i, :1])  

        # if self.config['is_harma']:
        #     if self.align_before:
        #         img_emb,feas_vis = self.get_vis_emb(image)
        #         txt_emb,feas_txt = self.get_txt_emb(text_ids)
        #     else:

        #         img_emb = self.get_vis_emb(image)
        #         txt_emb=self.get_txt_emb(text_ids)
        #     # txt_emb = self.get_text_embeds(text_ids)
        # else:
        #     img_emb= self.get_vision_fusion_embeds(image, self.config)
        #     txt_emb = self.get_text_fusion_embeds(text_ids, self.config)

        if self.use_affil_loss:
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            loss_affil = self.get_affil_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            return loss_contr, loss_affil
        elif self.use_triplet_loss:
            loss_triplet = self.get_triplet_loss(img_emb, txt_emb)
            return loss_triplet
        else:
            loss_before_contr = []
            # if self.align_before:
            #     for i in range(len(feas_vis)):
            #         # print("vis",feas_vis[i].shape)
            #         loss_contr = self.get_contr_loss(feas_vis[i],feas_txt[i], idx=idx, label=label, config=self.config)
            #         loss_before_contr.append(loss_contr)
            #     total_loss_before = sum(loss_before_contr)
            # loss_triplet = self.weighted_triplet_loss(img_emb, txt_emb)
            # if self.align_before:
            #     return loss_contr,loss_triplet,total_loss_before
            loss_contr = self.get_contr_loss(
            global_img_feat, global_txt_feat, idx)
            # loss_triplet = self.get_triplet_loss(
            # global_img_feat, global_txt_feat)
            # loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            loss_triplet = self.weighted_triplet_loss(global_img_feat, global_txt_feat)
            # loss_fine  =  self.get_fine_grained_loss(patch_feats, noun_feats_list)
            #TODO new loss
            return loss_contr,loss_triplet