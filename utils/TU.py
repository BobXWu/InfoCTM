import numpy as np
import argparse
from sklearn.feature_extraction.text import CountVectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    args = parser.parse_args()
    return args


def TU_eva(texts):
    K = len(texts)
    T = len(texts[0].split())
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
    counter = vectorizer.fit_transform(texts).toarray()

    TU = 0.0
    TF = counter.sum(axis=0)
    cnt = TF * (counter > 0)

    for i in range(K):
        TU += (1 / cnt[i][np.where(cnt[i] > 0)]).sum() / T
    TU /= K

    return TU


if __name__ == "__main__":
    args = parse_args()
    texts = list()
    with open(args.data_path, 'r') as file:
        for line in file:
            texts.append(line.strip())

    TU = TU_eva(texts)

    print(f"===>TU: {TU:.5f}")
