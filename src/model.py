"""阶段3：模型架构
- EncoderCNN: 预训练 ResNet-50，输出 14x14x2048 空间特征图
- Attention: Bahdanau 加性注意力（软注意力）
- DecoderWithAttention: LSTMCell 逐步解码，每步对图像区域加权（Show, Attend and Tell, Xu et al. 2015）

公式（写报告用）:
  e_ti = f_att(a_i, h_{t-1}) = w^T relu(W_a a_i + W_h h_{t-1})
  alpha_ti = softmax(e_ti)
  z_t = beta_t * sum_i alpha_ti * a_i,  beta_t = sigmoid(W_b h_{t-1})  (门控)
  h_t, c_t = LSTM([Ey_{t-1}; z_t], h_{t-1}, c_{t-1})
  p(y_t) = softmax(W_o dropout(h_t))
损失 = 交叉熵 + alpha_c * sum_i (1 - sum_t alpha_ti)^2   (双重随机注意力正则)
"""
import torch
from torch import nn
import torchvision


class EncoderCNN(nn.Module):
    def __init__(self, encoded_size=14):
        super().__init__()
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去掉avgpool和fc
        self.pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))
        self.fine_tune(False)

    def forward(self, x):
        f = self.resnet(x)                      # (B, 2048, H/32, W/32)
        f = self.pool(f)                        # (B, 2048, 14, 14)
        f = f.permute(0, 2, 3, 1)               # (B, 14, 14, 2048)
        return f.reshape(f.size(0), -1, f.size(-1))  # (B, 196, 2048)

    def fine_tune(self, on=True):
        """默认冻结全部；开启后仅微调 layer3/layer4（小显存友好）"""
        for p in self.resnet.parameters():
            p.requires_grad = False
        if on:
            for block in list(self.resnet.children())[6:]:
                for p in block.parameters():
                    p.requires_grad = True


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, h):
        # encoder_out: (B, 196, enc_dim), h: (B, dec_dim)
        att = self.full_att(torch.relu(
            self.enc_att(encoder_out) + self.dec_att(h).unsqueeze(1))).squeeze(2)  # (B,196)
        alpha = torch.softmax(att, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, enc_dim)
        return context, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, decoder_dim=512,
                 attention_dim=512, encoder_dim=2048, dropout=0.5, use_attention=True):
        super().__init__()
        self.vocab_size, self.encoder_dim, self.decoder_dim = vocab_size, encoder_dim, decoder_dim
        self.use_attention = use_attention  # False = 纯CNN+LSTM基线（每步用全图均值特征）
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # 注意力门控
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)

    def init_hidden(self, encoder_out):
        mean = encoder_out.mean(dim=1)
        return self.init_h(mean), self.init_c(mean)

    def compute_context(self, encoder_out, h):
        """注意力版：按 h 加权图像区域；基线版：每步用全图均值特征（无注意力）。
        返回 (context, alpha)。基线版 alpha 为均匀分布（仅占位，不参与正则）。"""
        if self.use_attention:
            return self.attention(encoder_out, h)
        context = encoder_out.mean(dim=1)
        alpha = encoder_out.new_full((encoder_out.size(0), encoder_out.size(1)),
                                     1.0 / encoder_out.size(1))
        return context, alpha

    def step(self, emb, context, h, c):
        gate = torch.sigmoid(self.f_beta(h))
        h, c = self.lstm(torch.cat([emb, gate * context], dim=1), (h, c))
        return self.fc(self.dropout(h)), h, c

    def forward(self, encoder_out, captions, lengths):
        """训练用 teacher forcing。captions 已按长度降序排序。
        返回 predictions (B, T, V), alphas (B, T, 196), decode_lengths
        """
        B = encoder_out.size(0)
        h, c = self.init_hidden(encoder_out)
        embeds = self.embedding(captions)               # (B, L, E)
        decode_lengths = (lengths - 1).tolist()         # 不预测<start>之后的pad
        T = max(decode_lengths)
        preds = encoder_out.new_zeros(B, T, self.vocab_size)
        alphas = encoder_out.new_zeros(B, T, encoder_out.size(1))
        for t in range(T):
            bt = sum(l > t for l in decode_lengths)     # 仍在解码的样本数（已排序）
            context, alpha = self.compute_context(encoder_out[:bt], h[:bt])
            logit, h_new, c_new = self.step(embeds[:bt, t], context, h[:bt], c[:bt])
            h = torch.cat([h_new, h[bt:]]) if bt < B else h_new
            c = torch.cat([c_new, c[bt:]]) if bt < B else c_new
            preds[:bt, t] = logit
            alphas[:bt, t] = alpha
        return preds, alphas, decode_lengths

    @torch.no_grad()
    def sample_greedy(self, encoder_out, start_idx, end_idx, max_len=50):
        """批量贪心解码（验证期间快速算BLEU用）。返回 (B, <=max_len) 的id序列"""
        B = encoder_out.size(0)
        h, c = self.init_hidden(encoder_out)
        words = torch.full((B,), start_idx, dtype=torch.long, device=encoder_out.device)
        done = torch.zeros(B, dtype=torch.bool, device=encoder_out.device)
        seqs = []
        for _ in range(max_len):
            context, _ = self.compute_context(encoder_out, h)
            logit, h, c = self.step(self.embedding(words), context, h, c)
            words = logit.argmax(dim=1)
            words[done] = end_idx
            seqs.append(words.clone())
            done |= words.eq(end_idx)
            if done.all():
                break
        return torch.stack(seqs, dim=1)


def count_params(*modules):
    total = sum(p.numel() for m in modules for p in m.parameters())
    trainable = sum(p.numel() for m in modules for p in m.parameters() if p.requires_grad)
    return total, trainable
