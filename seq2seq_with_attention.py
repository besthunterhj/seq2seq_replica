import copy
import random
import re
import unicodedata
from collections import Counter
from typing import List, Callable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.vocab import vocab as vc
from tqdm import tqdm

device = torch.device("mps")
MAX_LEN = 10
CLIP = 0.1
# test the ability of the attention mechanism


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


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p: List[str]):
    return len(p[0].split(' ')) < MAX_LEN and \
        len(p[1].split(' ')) < MAX_LEN and \
        p[1].startswith(eng_prefixes)


def filter_pairs(pairs: List[List[str]]):
    return [pair for pair in pairs if filter_pair(pair)]


class TranslationDataset(Dataset):

    def __init__(self, path: str):
        self.fra_seqs, self.eng_seqs = self.read_corpus(path=path)

    def __getitem__(self, index):
        return self.fra_seqs[index], self.eng_seqs[index]

    def __len__(self):
        return len(self.eng_seqs)

    @classmethod
    def read_corpus(cls, path: str):

        eng_seqs = []
        fra_seqs = []
        pairs = []

        with open(path, "r", encoding="utf-8") as f_obj:
            lines = f_obj.readlines()

            for line in lines:
                current_pair = [normalize_string(s) for s in line.strip().split("\t")]
                current_pair.reverse()
                pairs.append(current_pair)

        pairs = filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))

        for pair in pairs:
            fra_seqs.append(pair[0])
            eng_seqs.append(pair[-1])

        return fra_seqs, eng_seqs


def create_vocab(texts: list, tokenizer: Callable, min_freq: int, unknown_token: str = "<pad>",
                 unknown_index: int = 0) -> Vocab:

    all_tokens = [
        token
        for text in texts
        for token in tokenizer(text)
    ]

    tokens_counter = Counter(all_tokens)

    tokens_dict = dict(
        sorted(
            tokens_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    vocab = vc(
        ordered_dict=tokens_dict,
        min_freq=min_freq,
        specials=["<sos>", "<eos>"]
    )

    vocab.insert_token(
        token=unknown_token,
        index=unknown_index
    )

    # let the vocabulary ignore the unknown word
    vocab.set_default_index(index=unknown_index)

    return vocab


def pad_for_encoder(token_indexes: List[int], max_len: int, vocab: Vocab, default_padding_val: int = 0) -> List[int]:
    """
    Normalize the length of current list of token_indexes, so it can change to part of tensor
    :param vocab: the vocabulary for looking up the indexes of special characters
    :param default_padding_val: the default index of the character <pad>
    :param token_indexes: the indexes for tokens of current text
    :param max_len: the maximum length of normalization
    :return: List[int], a sequence which be normalized for its length
    """

    eos_index = vocab.get_stoi()["<eos>"]

    if len(token_indexes) > max_len:
        cut_seq = token_indexes[:max_len]

        # add the <eos> character
        cut_seq.append(eos_index)
        return cut_seq

    else:
        padded_token_indexes = token_indexes.copy()
        for i in range(max_len - len(token_indexes)):
            padded_token_indexes.append(default_padding_val)

        # add the <eos> character
        padded_token_indexes.append(eos_index)

        return padded_token_indexes


def pad_for_decoder(token_indexes: List[int], max_len: int, vocab: Vocab, default_padding_val: int = 0) -> List[int]:
    """
    Normalize the length of current list of token_indexes, so it can change to part of tensor
    :param vocab: the vocabulary for looking up the indexes of special characters
    :param default_padding_val: the default index of the character <pad>
    :param token_indexes: the indexes for tokens of current text
    :param max_len: the maximum length of normalization
    :return: List[int], a sequence which be normalized for its length
    """

    sos_index = vocab.get_stoi()["<sos>"]

    if len(token_indexes) > max_len:
        cut_seq = token_indexes[:max_len]

        # add the <eos> character
        cut_seq.insert(0, sos_index)
        return cut_seq

    else:
        padded_token_indexes = token_indexes.copy()
        for i in range(max_len - len(token_indexes)):
            padded_token_indexes.append(default_padding_val)

        # add the <eos> character
        padded_token_indexes.insert(0, sos_index)

        return padded_token_indexes


def collate_func(samples: List[Tuple[str, str]], fra_tokenizer: Callable, eng_tokenizer: Callable,
                 fra_vocab: Vocab, eng_vocab: Vocab, max_len: int) -> dict:
    """
     zip(*parameter): the "parameter" must be a list of tuples, and this function is separate it to two list which
     consists of the parameter[i][0] and parameter[i][-1]
    """
    fra_seqs, eng_seqs = list(zip(*samples))

    fra_seqs_tensor = torch.tensor(list(
        map(
            lambda current_sequence: pad_for_encoder(fra_vocab(fra_tokenizer(current_sequence)),
                                                     max_len=max_len, vocab=fra_vocab),
            fra_seqs
        )
    ))

    eng_seqs_tensor = torch.tensor(list(
        map(
            lambda current_sequence: pad_for_decoder(eng_vocab(eng_tokenizer(current_sequence)),
                                                     max_len=max_len, vocab=eng_vocab),
            eng_seqs
        )
    ))

    # Recommend to return the dictionary
    return {
        "fras": fra_seqs_tensor,
        "engs": eng_seqs_tensor
    }


class Encoder(nn.Module):

    def __init__(self, word_num: int, embedding_size: int, hidden_size: int, decoder_hidden_size: int, dropout: float):
        super(Encoder, self).__init__()
        # inherent attributes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # model architecture
        self.embedding_layer = nn.Embedding(num_embeddings=word_num, embedding_dim=self.embedding_size)
        self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=2 * self.hidden_size, out_features=decoder_hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    # src: [batch_size, sequence_len]
    def forward(self, src: torch.Tensor):

        # embedded: [batch_size, sequence_len, embedding_size]
        embedded = self.dropout(self.embedding_layer(src))

        # outputs: [batch_size, sequence_len, hidden_size * 2], hidden: [2, batch_size, hidden_size]
        outputs, hidden = self.rnn(embedded)

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        # initial decoder hidden is final hidden state of the forwards and backwards
        # the hidden state should be activated before sent to the decoder
        # hidden: [batch_size, decoder_hidden_size]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int):
        super(Attention, self).__init__()

        # model architecture
        # attn is the Linear layer that map the features from encoder_outputs and encoder_hn to energy features(Et)
        self.attn = nn.Linear(in_features=(encoder_hidden_size * 2 + decoder_hidden_size), out_features=decoder_hidden_size)
        # v is the Linear that maps the features to attn scores, letting the attn variable to [batch_size, sequence_len]
        self.v = nn.Linear(in_features=decoder_hidden_size, out_features=1, bias=False)

    def forward(self, encoder_outputs: torch.Tensor, hidden: torch.Tensor):
        # encoder_outputs: [batch_size, sequence_len, 2 * encoder_hidden_size]
        # hidden: [batch_size, decoder_hidden_size]

        # record the values of batch_size and sequence_len
        batch_size = encoder_outputs.shape[0]
        sequence_len = encoder_outputs.shape[1]

        # let the dimensions of the variable "hidden" to [batch_size, sequence_len, decoder_hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, sequence_len, 1)

        # calculate the energy variable
        # torch.cat((hidden, encoder_outputs), dim=2) ->: [batch_size, sequence_len, decoder_size + 2 * encoder_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy: [batch_size, sequence_len, decoder_hidden_size]
        attention_scores = self.v(energy).squeeze(2)

        # attention_scores: [batch_size, sequence_len]

        return F.softmax(attention_scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, word_num: int, embedding_size: int, enc_hid: int, dec_hid: int, attention: Attention, dropout: float):
        super(Decoder, self).__init__()
        # inherent attributes
        self.output_dim = word_num
        self.embedding_size = embedding_size
        self.attention = attention

        # model architecture
        self.embedding_layer = nn.Embedding(num_embeddings=word_num, embedding_dim=self.embedding_size)
        self.rnn = nn.GRU(
            input_size=(2 * enc_hid) + self.embedding_size,
            hidden_size=dec_hid,
            )

        self.fc_out = nn.Linear(in_features=dec_hid + (2 * enc_hid) + self.embedding_size, out_features=self.output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X: torch.Tensor, encoder_outputs: torch.Tensor, hidden: torch.Tensor):
        # X: [batch_size, ]
        # encoder_outputs: [batch_size, sequence_len, 2 * encoder_hidden_size]
        # hidden: [batch_size, decoder_hidden_size]

        # let the first dimension(which means: sequence_len) of X become 1
        X = X.unsqueeze(0)

        # X: [1, batch_size]; embedded: [1, batch_size, embedding_size]
        embedded = self.dropout(self.embedding_layer(X))

        # initialize the attention scores
        # attention_scores: [batch_size, sequence_len]
        attention_scores = self.attention(encoder_outputs, hidden)

        # in order to concatenate attention_scores and encoder_outputs, we need to change the dimensions of the prior
        # attention_scores: [batch_size, 1, sequence_len]
        attention_scores = attention_scores.unsqueeze(1)

        # the variable "weighted" stores the weighted sum of the encoder_outputs
        # And "weighted" can guide the "embedded" to get information from some relevant words come from source sentence
        # weighted: [batch_size, 1, 2 * encoder_hidden_size]
        weighted = torch.bmm(attention_scores, encoder_outputs)
        # weighted: [1, batch_size, 2 * encoder_hidden_size]
        weighted = weighted.permute(1, 0, 2)

        # concatenate the embedded and weighted
        rnn_input = torch.cat((embedded, weighted), dim=2)

        # generate the output and hidden state of Decoder RNN
        # dec_output: [1, batch_size, decoder_hidden_size]; dec_hidden: [1, batch_size, decoder_hidden_size]
        dec_output, dec_hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # remove the first dimension which equals 1
        dec_output = dec_output.squeeze(0)
        dec_hidden = dec_hidden.squeeze(0)
        embedded = embedded.squeeze(0)
        weighted = weighted.squeeze(0)

        # get the probability distribution of the words(target language)
        # prediction: [batch_size, self.output_dim(=word_num)]
        prediction = self.fc_out(torch.cat((dec_output, embedded, weighted), dim=1))

        return prediction, dec_hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super(Seq2Seq, self).__init__()
        # inherent attributes
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_sequence, trg_sequence, teacher_forcing_ratio: float = 0.5):
        # input_sequence, trg_sequence: [batch_size, sequence_len]
        batch_size = trg_sequence.shape[0]
        sequence_len = trg_sequence.shape[1]
        trg_word_num = self.decoder.output_dim

        # init a tensor to store the predictions of Decoder
        outputs = torch.zeros(sequence_len, batch_size, trg_word_num).to(self.device)

        # input the sequence to encoder
        # encoder_outputs: [batch_size, sequence_len, hidden_size * 2], hidden: [batch_size, decoder_hidden_size]
        encoder_outputs, hidden = self.encoder(input_sequence)

        # in order to facilitate the prediction of Decoder, we change the order of dimensions of trg_sequence
        trg_sequence = trg_sequence.permute(1, 0)

        # the first input of the decoder must be <sos>
        decoder_input = trg_sequence[0, :]

        for t in range(1, sequence_len):
            decoder_current_output, decoder_hidden = self.decoder(decoder_input, encoder_outputs, hidden)

            # record the probability distribution of current step
            outputs[t] = decoder_current_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the predicted word of current step
            top1 = decoder_current_output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted word
            decoder_input = trg_sequence[t] if teacher_force else top1

        return outputs


# implement the version of weight initialization used in the paper
def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


# calculate the number of the parameters of the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model: Seq2Seq, criterion, optimizer, train_loader: DataLoader, clip):

    # sign the symbol of "training"
    model.train()

    # record the values of loss of each batch
    losses = []
    for batch in tqdm(train_loader):
        fra_batch = batch["fras"].to(device)
        eng_batch = batch["engs"].to(device)

        optimizer.zero_grad()

        # get the predictions of this batch(size: [sequence_len, batch_size, eng_word_num])
        # eng_batch (as well as the target batch): [batch_size, sequence_len]
        current_predictions = model(fra_batch, eng_batch)

        # change the dimensions of eng_batch to [sequence_len, batch_size]
        eng_batch = eng_batch.permute(1, 0)

        # get the size of the prediction distribution to calculate the loss
        pred_distribution_size = current_predictions.shape[-1]
        # remove the probability distribution of the first element of the sequence
        # and make the dimension to [true_sequence_len(10) * batch_size, eng_word_num]
        current_predictions = current_predictions[1:].view(-1, pred_distribution_size)
        # make the dimension of the target batch align to the dimension of current_predictions
        eng_batch = torch.reshape(eng_batch[1:], (-1, ))

        # current_predictions: [batch_size * (sequence_len - 1), eng_words_num]
        # eng_batch: [batch_size * (sequence_len - 1)]
        loss = criterion(current_predictions + 1e-8, eng_batch)
        losses.append(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    # calculate and print the training loss for this epoch
    train_loss = torch.tensor(losses).mean()
    print(f"Train Loss : {train_loss:.3f}")
    return train_loss


def predict(model, input_text: str, fra_tokenizer: Callable, fra_vocab: Vocab,
            eng_vocab: Vocab, sos_index: int, eos_index: int, pad_index: int):

    # get the tokens of the input sentence
    input_fra_tokens = fra_tokenizer(input_text)

    # change the textual tokens to their indexes
    input_fra_indexes = fra_vocab(input_fra_tokens)

    # padding for the input sentence
    input_fra_indexes.append(eos_index)

    # pack the input_fra_indexes to the tensor
    # temp_input: [batch_size = 1, sequence_len]
    temp_input = torch.tensor([input_fra_indexes]).to(device)

    # create a fake trg_sentence [batch_size = 1, MAX_LEN = 10]
    trg_sentence = [pad_index for i in range(MAX_LEN)]
    trg_sentence.insert(0, sos_index)
    trg_sentence = torch.tensor([trg_sentence]).to(device)

    # get the probability distribution of the prediction
    # current_prediction: [sequence_len, batch_size, eng_word_num]
    with torch.no_grad():
        current_prediction = model(temp_input, trg_sentence)

    current_prediction = current_prediction[1:]

    # current_prediction: [sequence_len, eng_word_num]
    current_prediction = current_prediction.squeeze(1)

    # results: [sequence_len, 1]
    results = torch.argmax(current_prediction, dim=1)
    print(results)

    translation_result = [eng_vocab.get_itos()[item] for item in results]
    print(translation_result)


def main(path: str, min_freq: int, embedding_size: int, enc_hidden: int, dec_hidden: int, lr: float, batch_size: int, epochs: int):
    # init the translation dataset
    translation_dataset = TranslationDataset(path=path)

    # init the tokenizers for French and English
    fra_tokenizer = get_tokenizer(None)
    eng_tokenizer = get_tokenizer("basic_english")

    # create the vocabularies for French and English
    fra_vocab = create_vocab(texts=translation_dataset.fra_seqs, tokenizer=fra_tokenizer, min_freq=min_freq)
    eng_vocab = create_vocab(texts=translation_dataset.eng_seqs, tokenizer=eng_tokenizer, min_freq=min_freq)

    # init the seq2seq model
    encoder = Encoder(
        word_num=len(fra_vocab),
        embedding_size=embedding_size,
        hidden_size=enc_hidden,
        decoder_hidden_size=dec_hidden,
        dropout=0.1,
    )

    attention = Attention(
        encoder_hidden_size=enc_hidden,
        decoder_hidden_size=dec_hidden,
    )

    decoder = Decoder(
        word_num=len(eng_vocab),
        embedding_size=embedding_size,
        enc_hid=enc_hidden,
        dec_hid=dec_hidden,
        attention=attention,
        dropout=0.1,
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device
    ).to(device)

    # init the weights of parameters of the model
    model.apply(init_weights)

    # print the number of the parameters which need to be learned
    print("The number of the trainable parameters is: ", count_parameters(model=model))

    # init the optimizer
    optimizer = Adam(params=model.parameters(), lr=lr)

    # init the criterion
    eng_word_mapping = eng_vocab.get_stoi()
    pad_index = eng_word_mapping["<pad>"]
    eos_index = eng_word_mapping["<eos>"]
    sos_index = eng_word_mapping["<sos>"]

    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    # init the collection function
    collate_fn = lambda samples: collate_func(
        samples=samples,
        fra_tokenizer=fra_tokenizer,
        eng_tokenizer=eng_tokenizer,
        fra_vocab=fra_vocab,
        eng_vocab=eng_vocab,
        max_len=MAX_LEN
    )

    # init the dataloader
    translation_dataloader = DataLoader(
        dataset=translation_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    # start to train
    saved_model = None
    for i in range(epochs):
        min_loss = float("inf")
        print("Epoch: ", (i + 1), "\n")

        current_loss = train(model=model, criterion=criterion, optimizer=optimizer, train_loader=translation_dataloader, clip=CLIP)

        if min_loss > current_loss:
            min_loss = current_loss
            print(min_loss)
            saved_model = copy.deepcopy(model)

    return saved_model

    # save the model
    # torch.save(model, "./seq2seq_attention_translation_model.pth")
    #
    # # load and test the model
    # loaded_model = torch.load("./seq2seq_attention_translation_model.pth")
    # predict(model=loaded_model,
    #         input_text="Va te !",
    #         fra_tokenizer=fra_tokenizer,
    #         fra_vocab=fra_vocab,
    #         eng_vocab=eng_vocab,
    #         sos_index=sos_index,
    #         eos_index=eos_index,
    #         pad_index=pad_index
    #         )


if __name__ == '__main__':

    seq2seq_model = main(
        path="./eng-fra.txt",
        min_freq=1,
        embedding_size=256,
        enc_hidden=512,
        dec_hidden=512,
        lr=1e-4,
        batch_size=128,
        epochs=10,
    )

    # save the model
    torch.save(seq2seq_model, "./seq2seq_attention_translation_model.pth")
