# 实验报告模板（Exp8 图像描述）— 按指导书第五部分结构

> 用法：跑完实验后，把【】中的内容替换为你的真实数据/截图，导出PDF（≤10页正文）。
> 每节后标注了"素材来源"，即对应的脚本输出文件。

---

## 标题页（1页）
- 实验名称：基于注意力机制的图像描述（Exp8）
- 姓名/学号/日期：【填】
- 实验环境摘要：Python 3.10, PyTorch 【版本】, GPU 【型号/显存】, Ubuntu/Windows 【版本】

## 摘要（0.5页）
本实验基于 COCO Captions 2014 数据集，实现了 CNN编码器（ResNet-50）+ Bahdanau软注意力 + LSTM解码器 的图像描述模型（Show, Attend and Tell 架构）。采用 teacher forcing 训练、束搜索（beam=3）推理。在【N】张测试图像上取得 BLEU-1=【】, BLEU-4=【】，并通过注意力热力图验证了模型生成每个词时对图像相关区域的定位能力。

## 1. 引言（0.5页）
- 任务背景：图像描述是连接计算机视觉与自然语言处理的跨模态任务……
- 相关工作：Show and Tell (Vinyals 2015，无注意力) → Show, Attend and Tell (Xu 2015，软注意力，本实验复现对象) → Transformer类方法
- 实验意义：完整实践编码器-解码器、注意力机制、BLEU评估三个核心知识点

## 2. 数据集分析与预处理（1-2页）
**素材来源：`python src/eda.py` 的终端输出 + figures/eda_*.png**

- 数据统计表：图像数【82783】、caption总数【】、每图caption数【~5】、caption平均长度【】、词汇量【】
- 插图1：caption长度分布直方图（figures/eda_caption_length.png）
- 插图2：高频词柱状图 或 样本网格图
- 难点分析：词频长尾分布（低频词→`<unk>`）、caption变长（padding+按长度排序解码）、一图多参考（多参考BLEU）
- 预处理：小写化+正则分词；词频≥5入词表（词表大小【】，覆盖率【】%）；序列加`<start>/<end>`、截断至52、batch内padding；图像 Resize(256)→RandomCrop(224)→水平翻转→色彩抖动→ImageNet归一化（验证集仅CenterCrop）
- 插图3（增强对比）：figures/augmentation_compare.png，展示同一图原图 vs 多个随机增强结果（对应 `python src/visualize_augmentation.py`）

## 3. 模型设计（1-2页）
**素材来源：src/model.py 顶部注释含全部公式；架构图自己画（draw.io 或手绘）**

- 架构图：图像→ResNet-50→14×14×2048特征图→(每个解码时间步)注意力加权→LSTMCell→词分布
- 公式：注意力打分 e_ti、softmax权重 α_ti、门控上下文 z_t、LSTM更新、输出层（抄model.py注释）
- 基线选择理由：**基线 = CNN编码器(ResNet-50) + LSTM解码器（无注意力，每步用全图均值特征）**，对应指导书Exp8基线模型；**改进 = 加入Bahdanau软注意力 + 束搜索解码**，对应进阶架构。报告需给出两者的BLEU对比（见第5节表）以体现"基线+至少一项改进"。
- 编码器冻结预训练ResNet-50（小数据量下防过拟合+省显存）
- 损失：交叉熵 + 双重随机注意力正则 λ·Σ(1−Σα)²（鼓励注意力覆盖全图；基线版无此项）
- 参数量：总计【】M，可训练【】M（train.py启动时打印）

## 4. 训练与调参（2-3页）
**素材来源：results/<run>/curves.png、results/experiments.csv、results/<run>/samples_epochN.txt**

- 训练配置：Adam lr=4e-4、batch=32、梯度裁剪5.0、ReduceLROnPlateau（监控BLEU-4）、早停patience=4
- 插图：基线 loss 曲线（train vs val 同图）+ BLEU-4 曲线
- 预测样例演化：贴 epoch2 / epoch6 / epoch12 的 samples_epochN.txt 对比，体现生成质量随训练提升
- 超参对比表（直接贴 experiments.csv，≥5组）：

| run | lr | batch | dim | dropout | finetune | best BLEU-4 | 训练时间 |
|---|---|---|---|---|---|---|---|
| baseline | 4e-4 | 32 | 512 | 0.5 | 否 | 【】 | 【】 |
| lr1e-3 | 1e-3 | ... | | | | | |
| ...（共6行） | | | | | | | |

- 调参分析（结合曲线写，例如）：lr=1e-3 时验证损失振荡→说明学习率偏大；dropout=0.3 时训练/验证损失差距增大→过拟合加重；微调编码器 BLEU 提升约【】但训练时间×【】
- 过拟合处理：数据增强 + dropout=0.5 + 早停 + 冻结编码器

## 5. 评估与可解释性（1-2页）
**素材来源：results/baseline/test_report_beam3.json、figures/attention_demo.png**

- 最终测试指标表：BLEU-1【】/ BLEU-2【】/ BLEU-3【】/ BLEU-4【】（beam=3，【N】张测试图）
- **基线 vs 改进对比**（核心，体现"基线+至少一项改进"）：

| 模型 | BLEU-1 | BLEU-4 | 说明 |
|---|---|---|---|
| 无注意力基线 (CNN+LSTM) | 【】 | 【】 | results/baseline_noattn/test_report_beam3.json |
| +注意力+束搜索 | 【】 | 【】 | results/attention/test_report_beam3.json |

预期注意力版 BLEU-4 高于基线，说明软注意力带来的增益。
- 与文献对比：Show, Attend and Tell 原文 COCO BLEU-4≈24.3（全量数据+更长训练）；本实验【】（子采样3万图，差距合理）
- 效率：参数量【】M、推理【】FPS、训练【】分钟
- 错误分析：贴 test_report 中 worst5 的2-3个样例，总结模式（如：物体计数错误/罕见场景描述泛化为常见模板"a man is standing..."/颜色混淆）
- 可解释性：贴注意力热力图（attention_demo.png），逐词分析（如生成"dog"时注意力集中在狗的区域）
- 可补充：beam=1 vs beam=3 vs beam=5 的BLEU对比（evaluate.py 换 --beam 跑三次）

## 6. AI 协作与版本管理（1页）
- GitHub 链接：【https://github.com/你的用户名/DeepLearning_Exp8_ImageCaptioning】
- 提交记录截图：【git log --oneline 截图，≥4个有意义提交】
- AI 对话示例（≥2个不同场景，按指导书模板）：
  - 场景1（架构设计）：提示词【"用PyTorch实现Show Attend and Tell的注意力LSTM解码器，编码器输出(B,196,2048)，要求支持变长caption的批量teacher forcing"】→ AI输出【贴关键代码段】→ 人工修改【如：调整了val/test划分种子、按自己显存改了默认batch size】
  - 场景2（调试/调参）：提示词【贴你真实的debug对话】→ AI输出【】→ 人工修改【】

## 7. 总结与反思（0.5页）
- 收获：注意力机制的实现细节（门控、双重随机正则）、变长序列批处理技巧、BLEU多参考评估
- 困难与解决：【如显存不足→减小batch/冻结编码器；BLEU初期为0→检查decode去除特殊token】
- 改进方向：Transformer解码器、CIDEr/METEOR指标、预提取特征加速训练、scheduled sampling

## 参考文献（0.5页）
1. Xu, K. et al. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. ICML 2015.
2. Vinyals, O. et al. Show and Tell: A Neural Image Caption Generator. CVPR 2015.
3. Lin, T.-Y. et al. Microsoft COCO: Common Objects in Context. ECCV 2014.
4. Papineni, K. et al. BLEU: a Method for Automatic Evaluation of Machine Translation. ACL 2002.
5. PyTorch 官方文档 https://pytorch.org/docs/

## 附录（不计页数）
- 核心代码 ≤30行：建议贴 model.py 中 Attention 类 + DecoderWithAttention.step
- 完整AI对话日志截图
