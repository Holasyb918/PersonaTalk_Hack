import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import DropPath
from timm.layers.helpers import to_2tuple

try:
    from .attention import Attention, Mlp, SinusoidalPosEmbedding
except:
    from attention import Attention, Mlp, SinusoidalPosEmbedding


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, tgt_mesh, ref_mesh, ref_tex):
        x = self.drop_path(self.attn(self.norm1_q(tgt_mesh), self.norm1_k(ref_mesh), self.norm1_v(ref_tex))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DualAttention(nn.Module):
    def __init__(self, d_model, n_head=1, d_in=256):
        super(DualAttention, self).__init__()
        self.qk_pos_emb = SinusoidalPosEmbedding(d_model)
        self.lip_attn = Block(d_model, n_head)
        self.face_attn = Block(d_model, n_head)

        self.norm_mesh_tgt = nn.LayerNorm(d_model)
        self.norm_lip_tex = nn.LayerNorm(d_model)
        self.norm_face_tex = nn.LayerNorm(d_model)


    def forward(self, mesh_tgt, lip_mesh_ref, lip_tex, face_mesh_ref, face_tex):
        mesh_tgt = self.qk_pos_emb(mesh_tgt)
        lip_mesh_ref = self.qk_pos_emb(lip_mesh_ref)
        face_mesh_ref = self.qk_pos_emb(face_mesh_ref)

        lip_tex = self.lip_attn(mesh_tgt, lip_mesh_ref, lip_tex)
        face_tex = self.face_attn(mesh_tgt, face_mesh_ref, face_tex)

        return lip_tex, face_tex

if __name__ == "__main__":
    d_model = 256
    n_head = 4
    model = DualAttention(d_model, n_head)
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params_num)
    mesh_tgt = torch.randn(1, 512, d_model)
    lip_mesh_ref = torch.randn(1, 512, d_model)
    lip_tex = torch.randn(1, 512, d_model)
    face_mesh_ref = torch.randn(1, 512, d_model)
    face_tex = torch.randn(1, 512, d_model)

    out_lip_tex, out_face_tex = model(mesh_tgt, lip_mesh_ref, lip_tex, face_mesh_ref, face_tex)
    print(out_lip_tex.shape, out_face_tex.shape)    
