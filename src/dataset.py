"""阶段2：COCO Captions 数据集与 DataLoader
- PairDataset: (图像, 单条caption) 对，用于训练/验证损失
- EvalDataset: 去重图像 + 全部参考caption，用于 BLEU 评估
"""
import json, os, pickle, random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from build_vocab import tokenize, Vocabulary  # noqa: F401

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_transform(train: bool):
    """训练用数据增强（随机裁剪+翻转+色彩抖动），验证/测试仅缩放归一化"""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_coco(ann_path):
    with open(ann_path) as f:
        d = json.load(f)
    id2file = {im["id"]: im["file_name"] for im in d["images"]}
    img2caps = {}
    for a in d["annotations"]:
        img2caps.setdefault(a["image_id"], []).append(a["caption"])
    return id2file, img2caps


def split_val_ids(img_ids, seed=42):
    """将 val2014 的图像 50/50 划分为 验证集/测试集（按 image_id，固定随机种子保证可复现）"""
    ids = sorted(img_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    half = len(ids) // 2
    return set(ids[:half]), set(ids[half:])


class PairDataset(Dataset):
    """每个样本 = (图像tensor, caption id序列, 序列长度)"""

    def __init__(self, img_dir, ann_path, vocab, train=True,
                 max_len=52, max_images=None, restrict_ids=None, seed=42):
        self.img_dir, self.vocab, self.max_len = img_dir, vocab, max_len
        self.transform = get_transform(train)
        id2file, img2caps = load_coco(ann_path)
        img_ids = list(img2caps.keys())
        if restrict_ids is not None:
            img_ids = [i for i in img_ids if i in restrict_ids]
        if max_images:  # 算力受限时子采样（指导书允许根据算力调整）
            rng = random.Random(seed)
            rng.shuffle(img_ids)
            img_ids = img_ids[:max_images]
        self.samples = []
        for iid in img_ids:
            for cap in img2caps[iid]:
                self.samples.append((id2file[iid], cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fname, cap = self.samples[i]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        img = self.transform(img)
        ids = self.vocab.encode(tokenize(cap), self.max_len)
        return img, torch.tensor(ids), len(ids)


def pair_collate(batch):
    """按长度降序排序并 padding（配合解码器按时间步递减批量解码）"""
    batch.sort(key=lambda x: x[2], reverse=True)
    imgs, caps, lens = zip(*batch)
    imgs = torch.stack(imgs)
    max_len = lens[0]
    padded = torch.zeros(len(caps), max_len, dtype=torch.long)
    for i, c in enumerate(caps):
        padded[i, : len(c)] = c
    return imgs, padded, torch.tensor(lens)


class EvalDataset(Dataset):
    """BLEU 评估：每个样本 = 一张图 + 其全部参考caption（分词后）"""

    def __init__(self, img_dir, ann_path, restrict_ids=None, max_images=None, seed=42):
        self.img_dir = img_dir
        self.transform = get_transform(train=False)
        id2file, img2caps = load_coco(ann_path)
        img_ids = sorted(img2caps.keys())
        if restrict_ids is not None:
            img_ids = [i for i in img_ids if i in restrict_ids]
        if max_images:
            rng = random.Random(seed)
            rng.shuffle(img_ids)
            img_ids = img_ids[:max_images]
        self.items = [(iid, id2file[iid], [tokenize(c) for c in img2caps[iid]]) for iid in img_ids]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        iid, fname, refs = self.items[i]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        return self.transform(img), iid, refs


def eval_collate(batch):
    imgs, iids, refs = zip(*batch)
    return torch.stack(imgs), list(iids), list(refs)


def load_vocab(path="data/vocab.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
