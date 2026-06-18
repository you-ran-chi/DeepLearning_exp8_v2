"""阶段2 代码要求：可视化数据增强前后的样本对比
对同一张图，左列显示原图（仅Resize+CenterCrop），右侧若干列显示训练增强
（RandomCrop+水平翻转+色彩抖动）的不同随机结果，直观展示增强效果。

用法: python src/visualize_augmentation.py
输出: figures/augmentation_compare.png
"""
import argparse, json, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import get_transform, IMAGENET_MEAN, IMAGENET_STD


def denorm(t):
    """反归一化回可显示的 [0,1] 图像"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main(data_root="data", n_images=4, n_aug=3, seed=0):
    with open(os.path.join(data_root, "annotations/captions_train2014.json")) as f:
        d = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    rng = random.Random(seed)
    ids = rng.sample(list(id2file), n_images)

    val_tf = get_transform(train=False)   # 原图基准（无增强）
    train_tf = get_transform(train=True)  # 训练增强

    fig, axes = plt.subplots(n_images, n_aug + 1, figsize=(3 * (n_aug + 1), 3 * n_images))
    axes = np.atleast_2d(axes)
    for r, iid in enumerate(ids):
        pil = Image.open(os.path.join(data_root, "train2014", id2file[iid])).convert("RGB")
        axes[r, 0].imshow(denorm(val_tf(pil)))
        axes[r, 0].set_title("原图(无增强)" if r == 0 else "", fontsize=10)
        for k in range(n_aug):
            torch.manual_seed(seed * 100 + r * 10 + k)  # 每格不同随机增强
            axes[r, k + 1].imshow(denorm(train_tf(pil)))
            axes[r, k + 1].set_title(f"增强样本{k+1}" if r == 0 else "", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    os.makedirs("figures", exist_ok=True)
    fig.tight_layout()
    fig.savefig("figures/augmentation_compare.png", dpi=150)
    print("已保存: figures/augmentation_compare.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    args = ap.parse_args()
    main(args.data_root)
