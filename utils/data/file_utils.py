import os
import yaml
import json
import argparse
import numpy as np


def print_topic_words(beta, vocab, num_top_word=15):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_word + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
    return topic_str_list


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def read_yaml(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config


def update_args(args, path):
    config = read_yaml(path)
    if config:
        args = vars(args)
        args.update(config)
        args = argparse.Namespace(**args)
    return args


def read_texts(path):
    texts = list()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def save_text(texts, path):
    with open(path, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text.strip() + '\n')
