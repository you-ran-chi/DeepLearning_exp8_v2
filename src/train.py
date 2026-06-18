"""阶段4-5：训练 + 可视化 + 超参记录
每个 run 一条命令，结果自动追加到 results/experiments.csv（调参对比表直接由此生成）。

基线示例:
  python src/train.py --run_name baseline --epochs 12 --batch_size 32 --lr 4e-4

调参示例（指导书要求至少5组）:
  python src/train.py --run_name lr1e-3      --lr 1e-3
  python src/train.py --run_name bs64        --batch_size 64
  python src/train.py --run_name drop0.3     --dropout 0.3
  python src/train.py --run_name dim256      --embed_dim 256 --decoder_dim 256 --attention_dim 256
  python src/train.py --run_name finetune    --fine_tune_encoder --encoder_lr 1e-5
"""
import argparse, csv, os, time
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu

from dataset import (PairDataset, EvalDataset, pair_collate, eval_collate,
                     load_coco, split_val_ids, load_vocab)
from model import EncoderCNN, DecoderWithAttention, count_params


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--run_name", default="baseline")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--decoder_dim", type=int, default=512)
    ap.add_argument("--attention_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--no_attention", action="store_true",
                    help="跑无注意力的纯CNN+LSTM基线（用于'基线 vs 注意力'对比）")
    ap.add_argument("--alpha_c", type=float, default=1.0, help="双重随机注意力正则系数")
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--fine_tune_encoder", action="store_true")
    ap.add_argument("--encoder_lr", type=float, default=1e-5)
    ap.add_argument("--max_train_images", type=int, default=30000,
                    help="训练图像子采样数（GTX1060可设20000~30000，全量约82783）")
    ap.add_argument("--bleu_val_images", type=int, default=1000, help="每轮验证BLEU用的图像数")
    ap.add_argument("--patience", type=int, default=4, help="早停轮数")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | run: {args.run_name}")

    vocab = load_vocab(os.path.join(args.data_root, "vocab.pkl"))
    train_img = os.path.join(args.data_root, "train2014")
    val_img = os.path.join(args.data_root, "val2014")
    train_ann = os.path.join(args.data_root, "annotations/captions_train2014.json")
    val_ann = os.path.join(args.data_root, "annotations/captions_val2014.json")

    # val2014 一分为二：验证集 / 测试集（测试集只在 evaluate.py 中使用一次）
    _, img2caps_val = load_coco(val_ann)
    val_ids, _test_ids = split_val_ids(img2caps_val.keys(), seed=args.seed)

    train_set = PairDataset(train_img, train_ann, vocab, train=True,
                            max_images=args.max_train_images, seed=args.seed)
    valloss_set = PairDataset(val_img, val_ann, vocab, train=False,
                              restrict_ids=val_ids, max_images=3000, seed=args.seed)
    bleu_set = EvalDataset(val_img, val_ann, restrict_ids=val_ids,
                           max_images=args.bleu_val_images, seed=args.seed)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True,
                              collate_fn=pair_collate, num_workers=args.num_workers, pin_memory=True)
    valloss_loader = DataLoader(valloss_set, args.batch_size, shuffle=False,
                                collate_fn=pair_collate, num_workers=args.num_workers)
    bleu_loader = DataLoader(bleu_set, args.batch_size, shuffle=False,
                             collate_fn=eval_collate, num_workers=args.num_workers)
    print(f"训练caption对: {len(train_set)} | 验证caption对: {len(valloss_set)} | BLEU验证图像: {len(bleu_set)}")

    encoder = EncoderCNN().to(device)
    decoder = DecoderWithAttention(len(vocab), args.embed_dim, args.decoder_dim,
                                   args.attention_dim, dropout=args.dropout,
                                   use_attention=not args.no_attention).to(device)
    if args.no_attention:
        args.alpha_c = 0.0  # 无注意力时双重随机正则无意义
    if args.fine_tune_encoder:
        encoder.fine_tune(True)

    total, trainable = count_params(encoder, decoder)
    print(f"参数量: 总计 {total/1e6:.1f}M | 可训练 {trainable/1e6:.1f}M")

    criterion = nn.CrossEntropyLoss()
    params = [{"params": decoder.parameters(), "lr": args.lr}]
    if args.fine_tune_encoder:
        params.append({"params": [p for p in encoder.parameters() if p.requires_grad],
                       "lr": args.encoder_lr})
    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)  # 监控BLEU-4

    run_dir = os.path.join("results", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join("logs", args.run_name))
    history = {"train_loss": [], "val_loss": [], "bleu4": []}
    best_bleu, bad_epochs, t0 = 0.0, 0, time.time()

    # 固定5张验证图，每2个epoch输出预测样例（指导书阶段4要求）
    fixed_imgs, fixed_ids, fixed_refs = next(iter(bleu_loader))
    fixed_imgs = fixed_imgs[:5].to(device)
    fixed_refs = fixed_refs[:5]

    def run_epoch(loader, train=True):
        encoder.train(train and args.fine_tune_encoder)
        decoder.train(train)
        total_loss, n = 0.0, 0
        for imgs, caps, lens in loader:
            imgs, caps, lens = imgs.to(device), caps.to(device), lens.to(device)
            with torch.set_grad_enabled(train):
                enc_out = encoder(imgs)
                preds, alphas, dec_lens = decoder(enc_out, caps, lens)
                targets = caps[:, 1:]  # 预测目标为下一个词
                p = pack_padded_sequence(preds, dec_lens, batch_first=True).data
                t = pack_padded_sequence(targets, dec_lens, batch_first=True).data
                loss = criterion(p, t)
                loss = loss + args.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
                    optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return total_loss / n

    @torch.no_grad()
    def eval_bleu():
        encoder.eval(); decoder.eval()
        refs_all, hyps_all = [], []
        for imgs, _, refs in bleu_loader:
            enc_out = encoder(imgs.to(device))
            seqs = decoder.sample_greedy(enc_out, vocab(vocab.START), vocab(vocab.END))
            for s, r in zip(seqs, refs):
                hyps_all.append(vocab.decode(s.tolist()).split())
                refs_all.append(r)
        return corpus_bleu(refs_all, hyps_all)  # BLEU-4

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, train=True)
        vl = run_epoch(valloss_loader, train=False)
        b4 = eval_bleu()
        scheduler.step(b4)
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["bleu4"].append(b4)
        writer.add_scalars("loss", {"train": tr, "val": vl}, epoch)
        writer.add_scalar("BLEU-4/val", b4, epoch)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:02d}/{args.epochs}] train {tr:.4f} | val {vl:.4f} | BLEU-4 {b4:.4f} | lr {lr_now:.1e}")

        if epoch % 2 == 0:  # 固定样本预测输出
            encoder.eval(); decoder.eval()
            with torch.no_grad():
                seqs = decoder.sample_greedy(encoder(fixed_imgs), vocab(vocab.START), vocab(vocab.END))
            lines = []
            for i, s in enumerate(seqs):
                lines.append(f"图{i} 预测: {vocab.decode(s.tolist())}")
                lines.append(f"图{i} 参考: {' '.join(fixed_refs[i][0])}")
            sample_txt = "\n".join(lines)
            with open(os.path.join(run_dir, f"samples_epoch{epoch}.txt"), "w") as f:
                f.write(sample_txt)
            writer.add_text("samples", sample_txt.replace("\n", "  \n"), epoch)

        if b4 > best_bleu:
            best_bleu, bad_epochs = b4, 0
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                        "args": vars(args), "epoch": epoch, "bleu4": b4},
                       os.path.join(run_dir, "best.pth"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"早停于 epoch {epoch}（BLEU-4 已 {args.patience} 轮未提升）")
                break

    train_minutes = (time.time() - t0) / 60
    # 损失/指标曲线图（报告必备）
    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["train_loss"], label="train loss")
    ax[0].plot(ep, history["val_loss"], label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss curves")
    ax[1].plot(ep, history["bleu4"], marker="o", color="green")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("BLEU-4"); ax[1].set_title("Validation BLEU-4")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "curves.png"), dpi=150)

    # 追加调参对比记录（阶段5表格直接来源）
    csv_path = "results/experiments.csv"
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "lr", "batch_size", "embed_dim", "decoder_dim", "dropout",
                        "fine_tune", "epochs_ran", "best_BLEU4", "train_min", "trainable_M"])
        w.writerow([args.run_name, args.lr, args.batch_size, args.embed_dim, args.decoder_dim,
                    args.dropout, args.fine_tune_encoder, len(history["train_loss"]),
                    f"{best_bleu:.4f}", f"{train_minutes:.1f}", f"{trainable/1e6:.1f}"])
    print(f"完成。最佳BLEU-4: {best_bleu:.4f} | 用时 {train_minutes:.1f} 分钟 | 已写入 {csv_path}")
    writer.close()


if __name__ == "__main__":
    main()
