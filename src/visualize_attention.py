"""阶段6可解释性：注意力权重可视化
生成caption时，逐词叠加显示模型关注图像的哪些区域（14x14 alpha上采样到原图）。

用法: python src/visualize_attention.py --ckpt results/baseline/best.pth \
        --image data/val2014/COCO_val2014_000000XXXXXX.jpg --out figures/attention_demo.png
"""
import argparse, math, os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dataset import get_transform, load_vocab
from model import EncoderCNN, DecoderWithAttention
from inference import beam_search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--out", default="figures/attention_demo.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(os.path.join(args.data_root, "vocab.pkl"))
    ckpt = torch.load(args.ckpt, map_location=device)
    margs = ckpt["args"]
    encoder = EncoderCNN().to(device).eval()
    decoder = DecoderWithAttention(len(vocab), margs["embed_dim"], margs["decoder_dim"],
                                   margs["attention_dim"],
                                   use_attention=margs.get("no_attention", False) is False).to(device).eval()
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    pil = Image.open(args.image).convert("RGB")
    img = get_transform(train=False)(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        seq, alphas = beam_search(decoder, encoder(img), vocab, beam_size=args.beam)
    words = vocab.decode(seq).split()
    print("生成caption:", " ".join(words))

    # 展示用：把原图center-crop成正方形再缩放，与alpha网格对齐
    side = min(pil.size)
    left, top = (pil.size[0] - side) // 2, (pil.size[1] - side) // 2
    disp = pil.crop((left, top, left + side, top + side)).resize((448, 448))
    disp = np.asarray(disp)

    T = len(words)
    cols = 5
    rows = math.ceil((T + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    axes.flat[0].imshow(disp); axes.flat[0].set_title("原图/original", fontsize=10)
    g = int(math.sqrt(alphas.size(1)))  # 14
    for t, w in enumerate(words):
        a = alphas[t].view(g, g).cpu().numpy()
        a = np.kron(a, np.ones((448 // g, 448 // g)))  # 最近邻上采样
        ax = axes.flat[t + 1]
        ax.imshow(disp)
        ax.imshow(a, alpha=0.6, cmap="jet")
        ax.set_title(w, fontsize=11)
    for ax in axes.flat:
        ax.axis("off")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
