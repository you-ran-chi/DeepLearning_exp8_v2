"""阶段2：构建词汇表（COCO Captions）
用法: python src/build_vocab.py --ann data/annotations/captions_train2014.json --min_freq 5
"""
import argparse, json, pickle, re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+")

def tokenize(s):
    return TOKEN_RE.findall(s.lower())

class Vocabulary:
    PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

    def __init__(self):
        self.word2idx, self.idx2word = {}, {}
        for w in [self.PAD, self.START, self.END, self.UNK]:
            self.add(w)

    def add(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = len(self.word2idx)
            self.idx2word[self.word2idx[w]] = w

    def __call__(self, w):
        return self.word2idx.get(w, self.word2idx[self.UNK])

    def __len__(self):
        return len(self.word2idx)

    def encode(self, tokens, max_len=None):
        ids = [self(self.START)] + [self(t) for t in tokens] + [self(self.END)]
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), self.UNK)
            if w == self.END:
                break
            if w not in (self.PAD, self.START):
                words.append(w)
        return " ".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/annotations/captions_train2014.json")
    ap.add_argument("--min_freq", type=int, default=5, help="词频阈值，低于此频次映射为<unk>")
    ap.add_argument("--out", default="data/vocab.pkl")
    args = ap.parse_args()

    with open(args.ann) as f:
        anns = json.load(f)["annotations"]

    counter = Counter()
    for a in anns:
        counter.update(tokenize(a["caption"]))

    vocab = Vocabulary()
    for w, c in counter.most_common():
        if c >= args.min_freq:
            vocab.add(w)

    with open(args.out, "wb") as f:
        pickle.dump(vocab, f)
    print(f"总词数: {len(counter)} | 阈值{args.min_freq}后词表大小: {len(vocab)} | 保存至 {args.out}")


if __name__ == "__main__":
    main()
