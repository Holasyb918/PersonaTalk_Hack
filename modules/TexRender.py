import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# encoder decoder
try:
    from .EncoderDecoder import AppearanceFeatureExtractor2D, TexDecoder
except:
    from EncoderDecoder import AppearanceFeatureExtractor2D, TexDecoder

# Dual Attention
try:
    from .DualAttnSampler import DualAttention
except:
    from DualAttnSampler import DualAttention


class TexRender(nn.Module):
    def __init__(self, image_channel, block_expansion, num_down_blocks, attn_dim=512):
        super(TexRender, self).__init__()
        self.encoder_mesh = AppearanceFeatureExtractor2D(
            image_channel, block_expansion, num_down_blocks, attn_dim
        )
        self.encoder_tex = AppearanceFeatureExtractor2D(
            image_channel, block_expansion, num_down_blocks, attn_dim
        )
        self.decoder = TexDecoder(
            image_channel, block_expansion, num_down_blocks, attn_dim
        )

        self.cross_attn = DualAttention(attn_dim)

    def pre_rearrange_for_encode(Self, x):
        return rearrange(x, "b n c h w -> (b n) c h w")

    def pre_rearrange_for_attn(Self, x, b, n):
        x = x.flatten(2).permute(0, 2, 1)
        lip_mesh_ref_feats = rearrange(x, "(b n) c d -> b (n c) d", b=b, n=n)
        return lip_mesh_ref_feats

    def post_rearrange_for_decode(Self, x, b, h, w):
        x = rearrange(x.permute(0, 2, 1), "b c (h w) -> b c h w", b=b, h=h, w=w)
        return x

    def forward(
        self, mesh_tgt, lip_mesh_ref, lip_tex, face_mesh_ref, face_tex, mask=None
    ):
        # mesh_tgt: Bx3x256x256
        # lip_mesh_ref: Bx5x3x256x256
        # lip_tex: Bx5x3x256x256
        # face_mesh_ref: Bx5x3x256x256
        # face_tex: Bx5x3x256x256
        bz, lt, c, h, w = lip_mesh_ref.shape
        bz, ft, c, h, w = lip_mesh_ref.shape

        # reshape
        lip_mesh_ref = self.pre_rearrange_for_encode(lip_mesh_ref)
        lip_tex = self.pre_rearrange_for_encode(lip_tex)
        face_mesh_ref = self.pre_rearrange_for_encode(face_mesh_ref)
        face_tex = self.pre_rearrange_for_encode(face_tex)

        # encode tgt mesh
        mesh_tgt_feats = self.encoder_mesh(mesh_tgt)
        bz, c, h, w = mesh_tgt_feats.shape
        mesh_tgt_feats = mesh_tgt_feats.flatten(2).permute(0, 2, 1)  # BxNxC -> BxCxN

        # encode ref lip mesh
        lip_mesh_ref_feats = self.encoder_mesh(lip_mesh_ref)
        lip_mesh_ref_feats = self.pre_rearrange_for_attn(lip_mesh_ref_feats, bz, lt)

        # encode ref face mesh
        face_mesh_ref_feats = self.encoder_mesh(face_mesh_ref)
        face_mesh_ref_feats = self.pre_rearrange_for_attn(face_mesh_ref_feats, bz, ft)

        # encode ref lip tex
        lip_tex_feats = self.encoder_tex(lip_tex)
        lip_tex_feats = self.pre_rearrange_for_attn(lip_tex_feats, bz, lt)

        # encode ref face tex
        face_tex_feats = self.encoder_tex(face_tex)
        face_tex_feats = self.pre_rearrange_for_attn(face_tex_feats, bz, ft)

        # cross attention
        lip_tex, face_tex = self.cross_attn(
            mesh_tgt_feats,
            lip_mesh_ref_feats,
            lip_tex_feats,
            face_mesh_ref_feats,
            face_tex_feats,
        )

        lip_tex_reshaped = self.post_rearrange_for_decode(lip_tex, bz, h, w)
        face_tex_reshaped = self.post_rearrange_for_decode(face_tex, bz, h, w)

        if mask is not None:
            mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=True)
            comp_feats = lip_tex_reshaped * mask + face_tex_reshaped * (1 - mask)
        else:
            comp_feats = lip_tex_reshaped + face_tex_reshaped

        mesh_tgt_downsampled = F.interpolate(
            mesh_tgt, size=(h, w), mode="bilinear", align_corners=True
        )
        # comp_feats = torch.cat([comp_feats, mesh_tgt_downsampled], dim=1)
        out = self.decoder(comp_feats, mesh_tgt_downsampled)

        return out


if __name__ == "__main__":
    mesh_tgt = torch.randn(1, 3, 256, 256)
    lip_mesh_ref = torch.randn(1, 5, 3, 256, 256)
    lip_tex = torch.randn(1, 5, 3, 256, 256)
    face_mesh_ref = torch.randn(1, 5, 3, 256, 256)
    face_tex = torch.randn(1, 5, 3, 256, 256)

    model = TexRender(3, 64, 2, 512)
    model_params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model_params_num)
    mask = torch.randn(1, 1, 64, 64)

    out = model(mesh_tgt, lip_mesh_ref, lip_tex, face_mesh_ref, face_tex, mask)
    print(out.shape)
