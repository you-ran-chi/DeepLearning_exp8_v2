"""阶段6：测试集最终评估
- 束搜索解码，BLEU-1/2/3/4
- 效率：参数量、推理FPS
- 输出最好/最差样例各5个（错误分析用）

用法: python src/evaluate.py --ckpt results/baseline/best.pth --beam 3 --max_images 2500
"""
import argparse, json, os, time
import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from dataset import EvalDataset, eval_collate, load_coco, split_val_ids, load_vocab
from model import EncoderCNN, DecoderWithAttention, count_params
from inference import beam_search


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--max_images", type=int, default=2500, help="测试图像数（全量约2万张较慢）")
    ap.add_argument("--seed", type=int, default=42)
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

    val_ann = os.path.join(args.data_root, "annotations/captions_val2014.json")
    _, img2caps = load_coco(val_ann)
    _val_ids, test_ids = split_val_ids(img2caps.keys(), seed=margs.get("seed", 42))
    test_set = EvalDataset(os.path.join(args.data_root, "val2014"), val_ann,
                           restrict_ids=test_ids, max_images=args.max_images, seed=args.seed)
    loader = DataLoader(test_set, batch_size=1, collate_fn=eval_collate, num_workers=2)
    print(f"测试图像数: {len(test_set)} | beam={args.beam}")

    smooth = SmoothingFunction().method1
    refs_all, hyps_all, per_image = [], [], []
    t0, n = time.time(), 0
    with torch.no_grad():
        for imgs, iids, refs in loader:
            enc_out = encoder(imgs.to(device))
            seq, _ = beam_search(decoder, enc_out, vocab, beam_size=args.beam)
            hyp = vocab.decode(seq).split()
            refs_all.append(refs[0]); hyps_all.append(hyp)
            s4 = sentence_bleu(refs[0], hyp, smoothing_function=smooth)
            per_image.append({"image_id": iids[0], "pred": " ".join(hyp),
                              "ref": " ".join(refs[0][0]), "bleu4": round(s4, 4)})
            n += 1
    elapsed = time.time() - t0

    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (1/3, 1/3, 1/3, 0), (0.25, 0.25, 0.25, 0.25)]
    bleu = [corpus_bleu(refs_all, hyps_all, weights=w) for w in weights]
    total, trainable = count_params(encoder, decoder)

    per_image.sort(key=lambda x: x["bleu4"])
    report = {
        "checkpoint": args.ckpt, "beam_size": args.beam, "num_test_images": n,
        "BLEU-1": round(bleu[0], 4), "BLEU-2": round(bleu[1], 4),
        "BLEU-3": round(bleu[2], 4), "BLEU-4": round(bleu[3], 4),
        "params_total_M": round(total / 1e6, 1), "params_trainable_M": round(trainable / 1e6, 1),
        "inference_FPS": round(n / elapsed, 2),
        "worst5": per_image[:5], "best5": per_image[-5:],
    }
    out = os.path.join(os.path.dirname(args.ckpt), f"test_report_beam{args.beam}.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)},
                     ensure_ascii=False, indent=2))
    print(f"完整报告（含最好/最差样例）已保存: {out}")


if __name__ == "__main__":
    main()
