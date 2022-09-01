import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import Multi30k

# test the ability of the attention mechanism

class Encoder(nn.Module):

    def __init__(self, word_num: int, embedding_size: int, hidden_size: int, decoder_hidden_size: int):
        super(Encoder, self).__init__()
        # inherent attributes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # model architecture
        self.embedding_layer = nn.Embedding(num_embeddings=word_num, embedding_dim=self.embedding_size)
        self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=2 * self.hidden_size, out_features=decoder_hidden_size)

    # src: [batch_size, sequence_len]
    def forward(self, src: torch.Tensor):

        # embedded: [batch_size, sequence_len, embedding_size]
        embedded = self.embedding_layer(src)

        # outputs: [batch_size, sequence_len, hidden_size * 2], hidden: [2, batch_size, hidden_size]
        outputs, hidden = self.rnn(embedded)

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        # initial decoder hidden is final hidden state of the forwards and backwards
        # the hidden state should be activated before sent to the decoder
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()


if __name__ == '__main__':
    input()
