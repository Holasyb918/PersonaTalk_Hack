import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class SinusoidalPosEmbedding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len=50000):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(SinusoidalPosEmbedding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.encoding = self.encoding.unsqueeze(0)
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len, _ = x.size()
        # [batch_size = 128, seq_len = 30]
        # print(x.shape)
        return self.encoding[:, :seq_len, :] + x
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        params_fc1_num = sum(p.numel() for p in self.fc1.parameters() if p.requires_grad)
        params_fc2_num = sum(p.numel() for p in self.fc2.parameters() if p.requires_grad)
        print(params_fc1_num, params_fc2_num, self.fc1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self, n_head):
        super(ScaleDotProductAttention, self).__init__()
        self.n_head = n_head
        self.softmax = nn.Softmax(dim=-1)
        if self.n_head > 1:
            self.forward = self.forward_mh
        else:
            self.forward = self.forward_sh

    def forward_sh(self, q, k, v, mask=None, e=1e-12):
        # single head
        batch_size, length, d_tensor = k.size()

        # 1. dot product with weight matrices
        k_t = k.transpose(1, 2)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

    def forward_mh(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class Attention(nn.Module):

    def __init__(self, d_model, n_head):
        super(Attention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention(n_head=n_head)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

        w_q_params = sum(p.numel() for p in self.w_q.parameters() if p.requires_grad)
        print('w_q_params', w_q_params)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        if self.n_head > 1:
            q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        if self.n_head > 1:
            out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


if __name__ == "__main__":
    length = 1024
    dim = 512
    mh_attn = Attention(dim, 4)
    inp_k = torch.randn(1, length, dim)
    inp_q = torch.randn(1, length, dim)
    inp_v = torch.randn(1, length, dim)
    out = mh_attn(inp_q, inp_k, inp_v)
    import time

    t0 = time.time()
    for i in range(1000):
        out = mh_attn(inp_q, inp_k, inp_v)
    t1 = time.time()
    print(t1 - t0)
    print(out.shape)

    sh_attn = Attention(dim, 1)
    inp_k = torch.randn(1, length, dim)
    # inp_q = torch.randn(1, length, dim)
    # inp_v = torch.randn(1, length, dim)
    inp_q = torch.randn(1, length, dim)
    inp_v = torch.randn(1, length, dim)
    out = sh_attn(inp_q, inp_k, inp_v)
    t0 = time.time()
    for i in range(1000):
        out = sh_attn(inp_q, inp_k, inp_v)
    t1 = time.time()
    print(t1 - t0)
    print(out.shape)
