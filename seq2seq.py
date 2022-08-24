import random
import re
import unicodedata
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


SOS_index = 0
EOS_index = 1


# init the Word2Index, Index2Word and count the words
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS, EOS and PAD

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicode_to_ascii(s: str):
    """
    Change the unicode string to its alternative of ascii
    :param s: the unicode string
    :return: ascii string
    """

    return "".join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s: str):
    """
    Normalize the ascii string, including lowercase, removing the non-letter characters
    :param s: the ascii string
    :return: normalized string
    """

    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_corpus(path: str, lang1: str, lang2: str):
    print("Reading Lines...")

    pairs = []
    with open(path, "r", encoding="utf-8") as f_obj:
        lines = f_obj.readlines()

        for line in lines:
            current_pair = [normalize_string(s) for s in line.strip().split("\t")]
            current_pair.reverse()
            pairs.append(current_pair)

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# select some simple and short sentences
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p: List[str]):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filter_pairs(pairs: List[List[str]]):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2):
    input_lang, output_lang, pairs = read_corpus("eng-fra.txt", lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# tokenization
def indexes_from_sentence(lang: Lang, sentence: str):
    return [lang.word2index[word] for word in sentence.split(' ')]


# transform tensor
def tensor_from_sentence(lang: Lang, sentence: str):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(1)
    # let each word has a bracket
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


# change the fra-eng pair to tensor
def tensors_from_pair(pairs, input_lang: Lang, output_lang: Lang, max_len: int):

    inputs = []
    targets = []
    for pair in pairs:
        current_input_indexes = indexes_from_sentence(input_lang, pair[0])
        current_target_indexes = indexes_from_sentence(output_lang, pair[1])

        if len(current_target_indexes) < max_len - 2:
            for i in range(max_len - len(current_target_indexes)):
                current_target_indexes.append(2)

        if len(current_input_indexes) < max_len:
            for i in range(max_len - len(current_input_indexes)):
                current_input_indexes.append(2)

        current_input_indexes.append(1)
        current_target_indexes.append(1)

        inputs.append(current_input_indexes)
        targets.append(current_target_indexes)

    input_tensor = torch.tensor(inputs)
    target_tensor = torch.tensor(targets)

    return input_tensor, target_tensor


class EncoderRNN(nn.Module):

    def __init__(self, word_num: int, embedding_size: int, hidden_size: int):
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_embeddings=word_num, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    # X: [batch_size, sequence_len]
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # embedded_X: [batch_size, sequence_len, embedding_size]
        embedded_X = self.embedding(X)

        # outputs: [batch_size, sequence_len, 2 * hidden_size]
        _, last_hidden_cell = self.lstm(embedded_X)

        return last_hidden_cell


class AtteDecoderRnn(nn.Module):

    def __init__(self, ):
        super(AtteDecoderRnn, self).__init__()


class DecoderRNN(nn.Module):

    def __init__(self, word_num: int, embedding_size: int, hidden_size: int):
        super(DecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_embeddings=word_num, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(in_features=2 * self.hidden_size, out_features=self.embedding_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # X: [batch_size, max_len + 1] (the first token is <sos>)
    def forward(self, X: torch.Tensor, previous_hidden: torch.Tensor):
        # embedded_X: [batch_size, max_len + 1, embedding_size]
        embedded_X = self.embedding(X)
        embedded_X = F.relu(embedded_X)

        # output: [batch_size, max_len + 1, 2 * hidden_size]; hidden is a tuple
        output, hidden = self.lstm(embedded_X, previous_hidden)
        output = self.softmax(self.fc(output))
        return output


def main(embedding_size: int, hidden_size: int):

    # load the data
    input_lang, output_lang, pairs = prepare_data('fra', 'eng')
    # print(input_lang.word2index)
    # print(output_lang.word2index)

    test_pairs = pairs[3421:3424]

    test_tuple = tensors_from_pair(test_pairs, input_lang=input_lang, output_lang=output_lang, max_len=7)

    decoder_inputs = torch.tensor([
        [0, 130, 79, 295, 529, 5],
        [0, 130, 79, 295, 529, 5],
        [0, 130, 79, 295, 529, 5]
                      ])

    encoder = EncoderRNN(word_num=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size)
    decoder = DecoderRNN(word_num=output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size)

    hidden_cell = encoder(test_tuple[0])
    outputs = decoder(decoder_inputs, hidden_cell)

    # 得到 outputs 后，将其转换为 [batch_size, sequence_len, 1] 其中 1 表示通过 max() 选取的当前最有可能 word 的 index
    # 再跟 targets([batch_size, sequence_len, 1]) 算 loss




if __name__ == '__main__':
    main(200, 50)
