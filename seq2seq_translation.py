import copy
import random
import re
import unicodedata
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# SOS_index = 0
# EOS_index = 1


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


class TranslationPairsDataset(Dataset):

    def __init__(self, pairs: List[List[str]]):
        super(TranslationPairsDataset, self).__init__()
        self.fras = pairs[0]
        self.engs = pairs[1]

    def __len__(self):
        return len(self.engs)

    def __getitem__(self, index: int):
        current_fra = self.fras[index]
        current_eng = self.engs[index]
        return current_fra, current_eng


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
    inputs_decoder = []
    targets = []

    for pair in pairs:
        current_input_indexes = indexes_from_sentence(input_lang, pair[0])
        current_target_indexes = indexes_from_sentence(output_lang, pair[1])

        if len(current_target_indexes) < max_len:
            for i in range(max_len - len(current_target_indexes)):
                current_target_indexes.append(2)

        if len(current_input_indexes) < max_len:
            for i in range(max_len - len(current_input_indexes)):
                current_input_indexes.append(2)

        current_input_decoder = copy.deepcopy(current_target_indexes)
        current_input_decoder.insert(0, 0)
        current_target_indexes.append(1)

        inputs.append(current_input_indexes)
        inputs_decoder.append(current_input_decoder)
        targets.append(current_target_indexes)

    return torch.tensor(inputs), torch.tensor(inputs_decoder), torch.tensor(targets)


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


# class AtteDecoderRnn(nn.Module):
#
#     def __init__(self, ):
#         super(AtteDecoderRnn, self).__init__()


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

        self.fc = nn.Linear(in_features=2 * self.hidden_size, out_features=word_num)

    # X: [batch_size, 1] (the first token is <sos>)
    def forward(self, X: torch.Tensor, previous_hidden_cell: Tuple[torch.Tensor, torch.Tensor]):
        X = X.unsqueeze(1)

        # embedded_X: [batch_size, 1, embedding_size]
        embedded_X = self.embedding(X)
        embedded_X = F.relu(embedded_X)

        # output: [batch_size, max_len + 1, 2 * hidden_size]; hidden is a tuple
        output, (hidden, cell) = self.lstm(embedded_X, previous_hidden_cell)

        # prediction: [batch_size, word_num]
        prediction = self.fc(output.squeeze(1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):

    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor,
                output_word_num: int, batch_size: int, max_len: int, teacher_force_ratio: float = 0.5):
        """
        the procedure of seq2seq model
        :param encoder_inputs: the size is [batch_size, max_len]
        :param decoder_inputs: the size is [batch_size, max_len + 1] (because the head of each sequence is <sos>)
        :param output_word_num: the number of the size of output language
        :param batch_size: the size of mini-batch
        :param max_len: the length of the input sequence of encoder (+1 to get the length of input sequence of decoder)
        :param teacher_force_ratio: the probability of using teacher_forcing
        :return:
        """

        # define the variable outputs to store all prediction of each step, outputs:[target_len, batch_size, char_num]
        outputs = torch.zeros(max_len + 1, batch_size, output_word_num)

        # get the last hidden state and cell state of the encoder
        hidden, cell = self.encoder(encoder_inputs)

        # the first input of the decoder must be <sos>
        decoder_input_n_step = decoder_inputs[:, 0]

        for i in range(max_len + 1):

            output, hidden, cell = self.decoder(decoder_input_n_step, (hidden, cell))

            # store the prediction at i-th step; output: [batch_size, output_word_num]
            outputs[i] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_force_ratio

            # get the highest predicted word from the prediction of this step
            prediction_n_step = output.argmax(1)

            # the final step doesn't need to update decoder_input_n_step
            if i < max_len:
                decoder_input_n_step = decoder_inputs[:, i + 1] if teacher_force else prediction_n_step

        # reshape the variable "outputs" to [batch_size, target_len, char_num]
        outputs = outputs.permute(1, 0, 2)
        return outputs


def main(embedding_size: int, hidden_size: int, epochs: int, batch_size: int, max_len: int = MAX_LENGTH):

    # load the data
    input_lang, output_lang, pairs = prepare_data('fra', 'eng')
    # print(input_lang.word2index)
    # print(output_lang.word2index)

    encoder = EncoderRNN(word_num=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size)
    decoder = DecoderRNN(word_num=output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size)
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    criterion = nn.CrossEntropyLoss(ignore_index=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)

    for epoch in range(epochs):
        # sign the training label
        model.train()

        optimizer.zero_grad()

        loss = 0
        for i in range(0, len(pairs), batch_size):

            if len(pairs) - i >= batch_size:
                end = i + batch_size
            else:
                break

            input_batch, input_decoder_batch, target_batch = tensors_from_pair(
                                                                pairs=pairs[i:end],
                                                                input_lang=input_lang,
                                                                output_lang=output_lang,
                                                                max_len=max_len)

            # outputs: [batch_size, max_len + 1, output_lang.n_words]
            outputs = model(input_batch, input_decoder_batch, output_lang.n_words, batch_size, max_len)

            for j in range(len(target_batch)):
                loss += criterion(outputs[j], target_batch[j])

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # save the model
    # torch.save(model, "./seq2seq_translation_model.pth")

    # test
    # model = torch.load("./seq2seq_translation_model.pth")

    # test batch
    # input_batch, decoder_input_batch, target_batch = tensors_from_pair(pairs=pairs[6666:6668], input_lang=input_lang, output_lang=output_lang, max_len=max_len)

    # predictions = model(input_batch, decoder_input_batch, output_lang.n_words, 2, max_len)

    # results = list(predictions.argmax(dim=2))

    # for item in results:
    #     translation = [output_lang.index2word[int(index)] for index in item]
    #     print(translation)

    # print()
    # print(pairs[6666:6668])
    # hidden_cell = encoder(test_tuple[0])
    # outputs = decoder(decoder_inputs, hidden_cell)

    # 得到 outputs 后，将其转换为 [batch_size, sequence_len, 1] 其中 1 表示通过 max() 选取的当前最有可能 word 的 index
    # 再跟 targets([batch_size, sequence_len, 1]) 算 loss




if __name__ == '__main__':
    main(200, 50, 30, 64)
