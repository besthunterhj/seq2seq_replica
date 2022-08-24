import random
from typing import List, Dict

import torch
import torch.nn as nn


def pad(word: str, max_len: int):

    if len(word) > max_len:
        padded_word = word[:max_len]
    else:
        padded_word = word + "P" * (max_len - len(word))

    return padded_word


def make_batch(sequences: List[List[str]], max_len: int, char_to_index: Dict):

    encoder_input_batch, decoder_input_batch, target_batch = [], [], []

    for sequence in sequences:
        # pad the current word if its length less than max_len
        for i in range(len(sequence)):
            sequence[i] = pad(sequence[i], max_len)

        encoder_input = [char_to_index[char] for char in sequence[0]]
        decoder_input = [char_to_index[char] for char in ("S" + sequence[-1])]
        target_output = [char_to_index[char] for char in (sequence[-1] + "E")]

        encoder_input_batch.append(encoder_input)
        decoder_input_batch.append(decoder_input)
        target_batch.append(target_output)

    return torch.tensor(encoder_input_batch), torch.tensor(decoder_input_batch), torch.tensor(target_batch)


class Encoder(nn.Module):

    def __init__(self, char_num: int, embedding_size: int, hidden_size: int):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding_layer = nn.Embedding(num_embeddings=char_num, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, encoder_input: torch.Tensor):
        # encoder_input: [batch_size, max_len]
        encoder_input_embedded = self.embedding_layer(encoder_input)

        # encoder_input_embedded: [batch_size, max_len, embedding_size]

        _, (last_hidden, last_cell) = self.lstm(encoder_input_embedded)

        return last_hidden, last_cell


class Decoder(nn.Module):

    def __init__(self, char_num: int, embedding_size: int, hidden_size: int):
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding_layer = nn.Embedding(num_embeddings=char_num, embedding_dim=self.embedding_size)
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(in_features=2 * hidden_size, out_features=char_num)

    def forward(self, decoder_input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        # decoder_input: [batch_size, 1]
        decoder_input = decoder_input.unsqueeze(1)
        decoder_input_embedded = self.embedding_layer(decoder_input)

        # decoder_input_embedded: [batch_size, 1, embedding_size]

        # output: [batch_size, 1, 2 * hidden_size]
        output, (last_hidden, last_cell) = self.lstm(decoder_input_embedded, (hidden, cell))

        # prediction: [batch_size, char_num]
        prediction = self.fc(output.squeeze(1))

        return prediction, last_hidden, last_cell


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    # encoder_input: [batch_size, max_len], decoder_input(for seq2seq model): [batch_size, max_len + 1]
    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor, char_num: int, teacher_forcing_ratio: float = 0.5):

        batch_size = decoder_input.shape[0]
        target_len = decoder_input.shape[1]

        # define the variable outputs to store all prediction of each step, outputs:[target_len, batch_size, char_num]
        outputs = torch.zeros(target_len, batch_size, char_num)

        # get the last hidden state and cell state of the encoder
        hidden, cell = self.encoder(encoder_input)

        # the first input of the decoder must be <sos>
        input_n_step = decoder_input[:, 0]

        for i in range(target_len):
            output, hidden, cell = self.decoder(input_n_step, hidden, cell)

            # store the prediction at i-th step; output: [batch_size, char_num]
            outputs[i] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if i < target_len - 1:
                input_n_step = decoder_input.T[i + 1] if teacher_force else top1

        # reshape the variable "outputs" to [batch_size, target_len, char_num]
        outputs = outputs.permute(1, 0, 2)
        return outputs


def make_testbatch(word: str, max_len: int, char_to_index: Dict):

    padded_word = pad(word=word, max_len=max_len)

    encoder_input_batch_test = [char_to_index[char] for char in padded_word]
    decoder_input_batch_test = [char_to_index[char] for char in 'S' + 'P' * max_len]

    return torch.tensor(encoder_input_batch_test).unsqueeze(0), torch.tensor(decoder_input_batch_test).unsqueeze(0)


def translate(word: str, max_len: int, char_to_index: Dict, index_to_char: Dict, model: Seq2Seq):

    encoder_input_batch_test, decoder_input_batch_test = make_testbatch(word=word, max_len=max_len, char_to_index=char_to_index)

    output = model(encoder_input_batch_test, decoder_input_batch_test, len(char_to_index.items()))

    # prediction: [batch_size, max_len]
    prediction = output.argmax(dim=2)[0]

    decoded = [index_to_char[int(index)] for index in prediction]
    end = decoded.index("E")
    result = ''.join(decoded[:end])

    return result.replace("P", "")


if __name__ == '__main__':
    # the number of the steps of the RNN, the length of sequence which less than will be padded
    max_len = 8
    # the size of the hidden state of the RNN
    hidden_size = 100
    embedding_size = 100

    alphabet = [char for char in "SEPabcdefghijklmnopqrstuvwxyz"]
    char_to_index = {n: i for i, n in enumerate(alphabet)}
    index_to_char = {i: n for i, n in enumerate(alphabet)}

    sequence_data = [
                        ["karen", "hikari"],
                        ["nana", "junna"],
                        ["maya", "claudine"],
                        ["futaba", "kaoruko"],
                        ["mahiro", "grass"]
    ]

    encoder_input_batch, decoder_input_batch, target_batch = make_batch(sequences=sequence_data, max_len=max_len, char_to_index=char_to_index)

    encoder = Encoder(char_num=len(alphabet), embedding_size=embedding_size, hidden_size=hidden_size)
    decoder = Decoder(char_num=len(alphabet), embedding_size=embedding_size, hidden_size=hidden_size)
    model = Seq2Seq(encoder=encoder, decoder=decoder)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train model
    for epoch in range(5000):
        # sign the training label
        model.train()

        optimizer.zero_grad()

        # outputs: [target_len, batch_size, char_num]
        outputs = model(encoder_input_batch, decoder_input_batch, len(alphabet))
        # outputs: [batch_size, target_len, char_num]

        loss = 0
        for i in range(len(target_batch)):
            loss += criterion(outputs[i], target_batch[i])

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # test
    test1 = translate("junon", max_len=max_len, char_to_index=char_to_index, index_to_char=index_to_char, model=model)
    print(test1)


