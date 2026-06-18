"""束搜索解码（单张图像），返回最优序列及逐词注意力图 alphas（可解释性分析用）"""
import torch


@torch.no_grad()
def beam_search(decoder, encoder_out, vocab, beam_size=3, max_len=50):
    """encoder_out: (1, 196, enc_dim)。返回 (word_ids列表, alphas: (T,196) tensor)"""
    device = encoder_out.device
    start, end = vocab(vocab.START), vocab(vocab.END)
    k = beam_size
    enc = encoder_out.expand(k, *encoder_out.shape[1:]).contiguous()  # (k,196,D)
    h, c = decoder.init_hidden(enc)
    seqs = torch.full((k, 1), start, dtype=torch.long, device=device)
    top_scores = torch.zeros(k, 1, device=device)
    seq_alphas = torch.ones(k, 1, enc.size(1), device=device)
    complete, complete_scores, complete_alphas = [], [], []

    for step in range(max_len):
        emb = decoder.embedding(seqs[:, -1])
        context, alpha = decoder.compute_context(enc, h)
        logit, h, c = decoder.step(emb, context, h, c)
        log_probs = torch.log_softmax(logit, dim=1)              # (k, V)
        scores = top_scores.expand_as(log_probs) + log_probs
        if step == 0:  # 第一步所有beam相同，只取第一行
            top_scores, top_words = scores[0].topk(k)
            prev_idx = torch.zeros(k, dtype=torch.long, device=device)
        else:
            top_scores, flat_idx = scores.view(-1).topk(k)
            V = log_probs.size(1)
            prev_idx, top_words = flat_idx // V, flat_idx % V

        seqs = torch.cat([seqs[prev_idx], top_words.unsqueeze(1)], dim=1)
        seq_alphas = torch.cat([seq_alphas[prev_idx], alpha[prev_idx].unsqueeze(1)], dim=1)
        h, c, enc = h[prev_idx], c[prev_idx], enc[prev_idx]

        is_end = top_words.eq(end)
        for i in torch.nonzero(is_end).flatten().tolist():
            complete.append(seqs[i].tolist())
            complete_scores.append(top_scores[i].item() / seqs.size(1))  # 长度归一化
            complete_alphas.append(seq_alphas[i])
        keep = ~is_end
        if keep.sum() == 0:
            break
        seqs, seq_alphas, h, c, enc = seqs[keep], seq_alphas[keep], h[keep], c[keep], enc[keep]
        top_scores = top_scores[keep].unsqueeze(1)
        k = keep.sum().item()

    if not complete:  # 达到max_len仍未结束
        complete = [seqs[0].tolist()]
        complete_scores = [top_scores[0].item() / seqs.size(1)]
        complete_alphas = [seq_alphas[0]]

    best = max(range(len(complete)), key=lambda i: complete_scores[i])
    return complete[best], complete_alphas[best][1:]  # 去掉<start>对应的占位alpha
