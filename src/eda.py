"""阶段1：探索性数据分析（EDA）
输出：
- figures/eda_caption_length.png  caption长度分布直方图
- figures/eda_top_words.png       高频词柱状图（去停用词）
- figures/eda_samples.png         3x3 样本图像网格（带caption）
- 终端打印数据统计表（直接抄进报告）

用法: python src/eda.py
"""
import json, os, random
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from build_vocab import tokenize

STOP = set("a an the of on in at with and is are to for by it this that there".split())


def main(data_root="data"):
    os.makedirs("figures", exist_ok=True)
    with open(os.path.join(data_root, "annotations/captions_train2014.json")) as f:
        d = json.load(f)
    images, anns = d["images"], d["annotations"]
    id2file = {im["id"]: im["file_name"] for im in images}

    lengths, counter = [], Counter()
    img2caps = {}
    for a in anns:
        toks = tokenize(a["caption"])
        lengths.append(len(toks))
        counter.update(toks)
        img2caps.setdefault(a["image_id"], []).append(a["caption"])

    lengths = np.array(lengths)
    caps_per_img = np.array([len(v) for v in img2caps.values()])
    sizes = [(im["width"], im["height"]) for im in images[:5000]]
    ws, hs = zip(*sizes)

    print("=" * 50)
    print("COCO Captions train2014 数据统计（报告表格用）")
    print("=" * 50)
    print(f"图像数量:            {len(images)}")
    print(f"caption总数:         {len(anns)}")
    print(f"每图caption数:       均值 {caps_per_img.mean():.2f}, 范围 [{caps_per_img.min()}, {caps_per_img.max()}]")
    print(f"caption长度(词):     均值 {lengths.mean():.1f}, 中位数 {np.median(lengths):.0f}, "
          f"P95 {np.percentile(lengths, 95):.0f}, 最大 {lengths.max()}")
    print(f"原始词汇量:          {len(counter)}")
    for th in (3, 5, 10):
        kept = sum(1 for c in counter.values() if c >= th)
        cover = sum(c for c in counter.values() if c >= th) / sum(counter.values())
        print(f"  词频>={th} 词表: {kept} 词, token覆盖率 {cover:.2%}")
    print(f"图像尺寸(前5000张):  宽 {min(ws)}~{max(ws)}, 高 {min(hs)}~{max(hs)}（需统一缩放）")
    print("难点提示: 词频呈长尾分布(类别不平衡)、caption长度差异大(需padding/截断)、"
          "同图多参考描述(评估需用多参考BLEU)")

    # 图1: caption长度分布
    plt.figure(figsize=(7, 4))
    plt.hist(lengths, bins=range(0, 40), edgecolor="white")
    plt.axvline(np.percentile(lengths, 95), color="red", ls="--", label="P95")
    plt.xlabel("caption length (words)"); plt.ylabel("count")
    plt.title("Caption Length Distribution (train2014)")
    plt.legend(); plt.tight_layout()
    plt.savefig("figures/eda_caption_length.png", dpi=150)

    # 图2: 高频词
    top = [(w, c) for w, c in counter.most_common(60) if w not in STOP][:20]
    plt.figure(figsize=(8, 4))
    plt.bar([w for w, _ in top], [c for _, c in top])
    plt.xticks(rotation=45, ha="right")
    plt.title("Top-20 Content Words"); plt.tight_layout()
    plt.savefig("figures/eda_top_words.png", dpi=150)

    # 图3: 样本网格
    rng = random.Random(0)
    sample_ids = rng.sample(list(img2caps.keys()), 9)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, iid in zip(axes.flat, sample_ids):
        p = os.path.join(data_root, "train2014", id2file[iid])
        if os.path.exists(p):
            ax.imshow(Image.open(p).convert("RGB"))
        cap = img2caps[iid][0]
        ax.set_title(cap[:55] + ("..." if len(cap) > 55 else ""), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("figures/eda_samples.png", dpi=150)
    print("已保存: figures/eda_caption_length.png, eda_top_words.png, eda_samples.png")


if __name__ == "__main__":
    main()
