# DeepLearning_Exp8 — 图像描述（Image Captioning）

基于 **CNN编码器（ResNet-50）+ 注意力LSTM解码器**（Show, Attend and Tell, Xu et al., ICML 2015）的图像描述模型，数据集为 **COCO Captions 2014**，评估指标 **BLEU-1~4**，含**注意力可视化**可解释性分析。

## 1. 环境配置

```bash
conda create -n exp8 python=3.10 -y
conda activate exp8
# GPU版（按自己CUDA版本调整，见 pytorch.org）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 2. 数据准备

从 https://cocodataset.org/#download 下载（国内可用镜像/迅雷）：

- `train2014.zip`（约13GB）
- `val2014.zip`（约6GB）
- `annotations_trainval2014.zip`

解压成如下结构：

```
data/
├── train2014/            # 82,783 张训练图
├── val2014/              # 40,504 张（代码内按 image_id 固定划分为 验证/测试 各半）
└── annotations/
    ├── captions_train2014.json
    └── captions_val2014.json
```

## 3. 运行流程（按顺序）

```bash
# 阶段1：EDA（输出统计表 + 3张图到 figures/）
python src/eda.py

# 阶段2：构建词表（词频>=5）
python src/build_vocab.py

# 阶段2：数据增强前后对比图（指导书明确要求）
python src/visualize_augmentation.py

# 阶段4：先跑无注意力基线（指导书Exp8的"基线模型"）
python src/train.py --run_name baseline_noattn --no_attention --epochs 12 --batch_size 32 --max_train_images 30000

# 阶段4：再跑注意力版（"进阶架构"，即至少一项改进）
python src/train.py --run_name attention --epochs 12 --batch_size 32 --lr 4e-4 --max_train_images 30000

# 阶段5：在注意力版基础上调参（至少5组，结果自动汇总到 results/experiments.csv）
python src/train.py --run_name lr1e-3   --lr 1e-3   --max_train_images 30000
python src/train.py --run_name bs64     --batch_size 64 --max_train_images 30000
python src/train.py --run_name drop0.3  --dropout 0.3 --max_train_images 30000
python src/train.py --run_name dim256   --embed_dim 256 --decoder_dim 256 --attention_dim 256 --max_train_images 30000
python src/train.py --run_name finetune --fine_tune_encoder --encoder_lr 1e-5 --max_train_images 30000

# 训练曲线实时查看
tensorboard --logdir logs

# 阶段6：测试集评估（束搜索，BLEU-1~4 + FPS + 最好/最差样例）
# 基线和注意力版各评估一次，做对比
python src/evaluate.py --ckpt results/baseline_noattn/best.pth --beam 3
python src/evaluate.py --ckpt results/attention/best.pth --beam 3

# 阶段6：注意力可视化（任选一张val图）
python src/visualize_attention.py --ckpt results/attention/best.pth \
    --image data/val2014/COCO_val2014_000000000074.jpg --out figures/attention_demo.png
```

## 4. 项目结构

```
src/
├── build_vocab.py          # 词表构建
├── dataset.py              # Dataset/DataLoader、数据增强
├── model.py                # 编码器/注意力/解码器（含公式注释）
├── inference.py            # 束搜索解码
├── train.py                # 训练+TensorBoard+曲线图+调参记录
├── evaluate.py             # 测试集BLEU/效率/错误分析
├── eda.py                  # 探索性数据分析
├── visualize_augmentation.py  # 数据增强前后对比
└── visualize_attention.py  # 注意力热力图
results/experiments.csv     # 超参对比表（自动生成）
figures/                    # 报告插图
```

## 5. 复现性

- 全部随机划分使用固定 seed=42（val/test划分、子采样）
- 每个 run 的超参保存在 checkpoint 内（`ckpt["args"]`）

## 6. 显存不足怎么办

- `--batch_size 16`，或 `--embed_dim 256 --decoder_dim 256 --attention_dim 256`
- 不开 `--fine_tune_encoder`（默认冻结 ResNet，仅占推理显存）
- 减小 `--max_train_images`
