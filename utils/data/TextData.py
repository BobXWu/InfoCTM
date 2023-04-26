import os
import numpy as np
import scipy
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from utils.data import file_utils


class BilingualTextDataset(Dataset):
    def __init__(self, bow_en, bow_cn):
        self.bow_en = bow_en
        self.bow_cn = bow_cn
        self.bow_size_en = len(self.bow_en)
        self.bow_size_cn = len(self.bow_cn)

    def __len__(self):
        return max(self.bow_size_en, self.bow_size_cn)

    def __getitem__(self, index):

        return_dict = {
            'bow_en': self.bow_en[(index % self.bow_size_en)],
            'bow_cn': self.bow_cn[(index % self.bow_size_cn)]
        }

        return return_dict


class DatasetHandler:
    def __init__(self, dataset, batch_size, lang1, lang2, dict_path):
        data_dir = f'../data/{dataset}'
        self.batch_size = batch_size

        self.train_texts_en, self.test_texts_en, self.train_bow_matrix_en, self.test_bow_matrix_en, self.vocab_en, self.word2id_en, self.id2word_en = self.read_data(data_dir, lang=lang1)
        self.train_texts_cn, self.test_texts_cn, self.train_bow_matrix_cn, self.test_bow_matrix_cn, self.vocab_cn, self.word2id_cn, self.id2word_cn = self.read_data(data_dir, lang=lang2)

        self.train_size_en = len(self.train_texts_en)
        self.train_size_cn = len(self.train_texts_cn)
        self.vocab_size_en = len(self.vocab_en)
        self.vocab_size_cn = len(self.vocab_cn)

        self.trans_dict, self.trans_matrix_en, self.trans_matrix_cn = self.parse_dictionary(dict_path)

        self.pretrain_word_embeddings_en = scipy.sparse.load_npz(os.path.join(data_dir, f'word2vec_{lang1}.npz')).toarray()
        self.pretrain_word_embeddings_cn = scipy.sparse.load_npz(os.path.join(data_dir, f'word2vec_{lang2}.npz')).toarray()

        self.train_bow_matrix_en, self.test_bow_matrix_en = self.move_to_cuda(self.train_bow_matrix_en, self.test_bow_matrix_en)
        self.train_bow_matrix_cn, self.test_bow_matrix_cn = self.move_to_cuda(self.train_bow_matrix_cn, self.test_bow_matrix_cn)

        self.train_loader = DataLoader(BilingualTextDataset(self.train_bow_matrix_en, self.train_bow_matrix_cn), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(BilingualTextDataset(self.test_bow_matrix_en, self.test_bow_matrix_cn), batch_size=batch_size, shuffle=False)

    def move_to_cuda(self, train_bow_matrix, test_bow_matrix):
        train_bow_matrix = torch.as_tensor(train_bow_matrix).float()
        test_bow_matrix = torch.as_tensor(test_bow_matrix).float()

        if torch.cuda.is_available():
            train_bow_matrix = train_bow_matrix.cuda()
            test_bow_matrix = test_bow_matrix.cuda()

        return train_bow_matrix, test_bow_matrix

    def read_data(self, data_dir, lang):
        train_texts = file_utils.read_texts(os.path.join(data_dir, 'train_texts_{}.txt'.format(lang)))
        test_texts = file_utils.read_texts(os.path.join(data_dir, 'test_texts_{}.txt'.format(lang)))
        vocab = file_utils.read_texts(os.path.join(data_dir, 'vocab_{}'.format(lang)))
        word2id = dict(zip(vocab, range(len(vocab))))
        id2word = dict(zip(range(len(vocab)), vocab))

        train_bow_matrix = scipy.sparse.load_npz(os.path.join(data_dir, 'train_bow_matrix_{}.npz'.format(lang))).toarray()
        test_bow_matrix = scipy.sparse.load_npz(os.path.join(data_dir, 'test_bow_matrix_{}.npz'.format(lang))).toarray()

        return train_texts, test_texts, train_bow_matrix, test_bow_matrix, vocab, word2id, id2word

    def parse_dictionary(self, dict_path):
        trans_dict = defaultdict(set)

        trans_matrix_en = np.zeros((self.vocab_size_en, self.vocab_size_cn), dtype='int32')
        trans_matrix_cn = np.zeros((self.vocab_size_cn, self.vocab_size_en), dtype='int32')

        dict_texts = file_utils.read_texts(dict_path)

        for line in dict_texts:
            terms = (line.strip()).split()
            if len(terms) == 2:
                cn_term = terms[0]
                en_term = terms[1]
                if cn_term in self.word2id_cn and en_term in self.word2id_en:
                    trans_dict[cn_term].add(en_term)
                    trans_dict[en_term].add(cn_term)
                    cn_term_id = self.word2id_cn[cn_term]
                    en_term_id = self.word2id_en[en_term]

                    trans_matrix_en[en_term_id][cn_term_id] = 1
                    trans_matrix_cn[cn_term_id][en_term_id] = 1

        return trans_dict, trans_matrix_en, trans_matrix_cn
