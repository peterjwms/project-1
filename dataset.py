"""
The dataset module contains any dataset representation as subclass of
``torch.utils.data.Dataset``.

Instruction:
---
To give you full flexibility to implement your preprocessing pipeline,
the only class provided is ``PDTBDataset`` which is required to put in
``torch.utils.data.DataLoader``.

Other than this class, you're free and welcome to implement any function
or class needed.
"""

import enum
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import to_level


class PDTBDataset(Dataset):
    """Dataset class for the PDTB dataset"""

    def __init__(
            self,
            data_path: str,
            # embeddings,
            # embedding_dim,
            level: int = 2,
            max_seq_len: int = 50
    ):
        # read in all the data from the PDTB files
        # do preprocessing and store the data
        # build vocabulary if needed - tokenidx mapping
        # then need it to be encoded as tensors
        self.data_path = data_path
        # self.embeddings = embeddings
        self.raw_data = []
        self.relations = []
        self.word2idx = {}
        self.idx2word = {}
        self.relation_indices = []
        self.labels = set()
        self.label2idx = {}
        self.idx2label = {}
        self.level = level
        self.max_seq_len = max_seq_len

        with open(data_path, "r") as f:
            for line in f:
                # TODO: probably can optimize this by not storing raw data -
                # hold onto it for now just to make sure I don't need any of the other pieces
                self.raw_data.append(json.loads(line))
                self.relations.append(
                    Relation(
                        arg1=self.raw_data[-1]["Arg1"]["RawText"],
                        arg2=self.raw_data[-1]["Arg2"]["RawText"],
                        connective=self.raw_data[-1]["Connective"]["RawText"],
                        sense=self.raw_data[-1]["Sense"][0],
                    )
                )
        self.build_vocab()
        self.build_label_map()
        self.num_classes = len(self.labels)
        # self.embedding_matrix = self.build_embedding_matrix(
        #     self.embeddings, embedding_dim=embedding_dim)

        # if embedding_type == "random":
        #     self.embedding_matrix = self.random_embeddings(
        #         len(self.word2idx), 300)
        # elif embedding_type == "glove":
        #     self.embedding_matrix = self.build_glove_embeddings(
        #         Path("dolma_300_2024_1.2M.100_combined.txt"), 300)

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        item = self.relation_indices[idx]
        # item = torch.tensor(self.relation_indices[idx])
        # turn these into one-hot vectors
        # use the one-hot vectors to get embeddings
        # return the embeddings and the label
        # sent_embedding = torch.zeros(
        #     (len(item), self.embedding_matrix.shape[1]))
        # for i, idx in enumerate(item):
        #     word_embedding = self.embedding_matrix[idx]
        #     sent_embedding[i] = word_embedding

        # handle the embeddings in the models instead
        # word_embeddings = torch.stack(
        #     [self.embedding_matrix[idx] for idx in item])
        # connective_embedding = word_embeddings[0]
        # avg_embedding = torch.mean(word_embeddings, dim=0)
        # avg_norm_embedding = avg_embedding / torch.norm(avg_embedding)
        # sum_embedding = torch.sum(embeddings, dim=0)

        # get the correct label at the desired level
        label = torch.tensor(self.label2idx[to_level(
            self.relations[idx].sense, self.level)])
        return torch.tensor(item), label
        # return avg_embedding + connective_embedding, label

    def build_vocab(self):
        # Implement vocabulary building here
        self.word2idx["<PAD>"] = len(self.word2idx)
        self.idx2word[len(self.idx2word)] = "<PAD>"
        self.word2idx["<UNK>"] = len(self.word2idx)
        self.idx2word[len(self.idx2word)] = "<UNK>"

        for item in self.relations:
            relation_idx = []
            for token in item.tokenize():
                # build vocab (token-idx mapping)
                if token not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[token] = idx
                    self.idx2word[idx] = token
                # build relation sequence as token indices
                relation_idx.append(self.word2idx[token])

            # truncate or pad each sequence to max_seq_len
            if len(relation_idx) > self.max_seq_len:
                relation_idx = relation_idx[:self.max_seq_len]
            else:
                relation_idx += [self.word2idx["<PAD>"]] * \
                    (self.max_seq_len - len(relation_idx))
            self.relation_indices.append(relation_idx)
            self.labels.add(to_level(item.sense, self.level))

    def build_label_map(self):
        for idx, label in enumerate(self.labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label

    # def build_embedding_matrix(self, embeddings, embedding_dim):
    #     # Create an embedding matrix
    #     embedding_matrix = []
    #     for idx, word in tqdm(self.idx2word.items(), desc="Building embedding matrix"):
    #         if word in embeddings:
    #             embedding_matrix.append(embeddings[word])
    #         else:
    #             # If the word is not found in GloVe, initialize it randomly
    #             # TODO: maybe some other way to do this?
    #             embedding_matrix.append(list(torch.randn(embedding_dim)))

    #     return torch.tensor(embedding_matrix)


class Relation:
    def __init__(self, arg1, arg2, connective, sense):
        self.arg1 = arg1
        self.arg2 = arg2
        self.connective = connective
        self.sense = sense

    def tokenize(self):
        self.arg1_toks = self.arg1.lower().strip().split()
        self.arg2_toks = self.arg2.lower().strip().split()
        self.connective_toks = self.connective.lower().strip().split()
        # TODO: test returning this tokenization in different orders (e.g., connective first/last)
        return self.connective_toks + self.arg1_toks + self.arg2_toks

    def __repr__(self):
        return f"Relation(arg1={self.arg1}, arg2={self.arg2}, connective={self.connective}, sense={self.sense})"


class GloveEmbeddingTypes(enum.Enum):
    glove300 = "glove\dolma_300_2024_1.2M.100_combined.txt"
    glove200 = "glove\wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
    glove100 = "glove\wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
    glove50 = "glove\wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"


def process_glove_file(
        glove_path: str,
) -> dict[str, list[float]]:
    """Builds a dictionary of GloVe embeddings from the specified file.
    Args:
        glove_path (str): Path to the GloVe embeddings file.

    Returns:
        dict[str, list[float]]: A dictionary mapping words to their GloVe embeddings.
    """
    # Load and process GloVe embeddings from the specified path
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe embeddings"):
            try:
                values = line.strip().split(' ')
                word = values[0]
                vector = list(map(float, values[1:]))
                if len(vector) != 50:
                    print(vector)
                embeddings[word] = vector
            except ValueError:
                continue  # Skip lines that don't conform to the expected format

    return embeddings


def build_glove_embeddings(
        vocab: dict[int, str],
        embeddings: dict[str, list[float]],
        embedding_dim: int
) -> torch.Tensor:
    """Builds an embedding matrix for the given vocabulary using GloVe embeddings.
    Args:
        vocab (dict[int, str]): A dictionary mapping token indices to words.
        embeddings (dict[str, list[float]]): A dictionary of GloVe embeddings.
        embedding_dim (int): The dimension of the embeddings.

    Returns: 
        torch.Tensor: The embedding matrix.
    """
    embedding_matrix = torch.zeros((len(vocab), embedding_dim))
    print(embedding_matrix.shape)
    for idx, word in tqdm(vocab.items(), desc="Building embedding matrix"):
        # print(idx, word)
        if word in embeddings:
            embedding_matrix[idx] = torch.tensor(
                embeddings[word], dtype=torch.float32)
        else:
            # If the word is not found in GloVe, initialize it randomly
            # TODO: maybe some other way to do this?
            embedding_matrix[idx] = torch.randn(embedding_dim)
    return embedding_matrix


# def build_random_embeddings(vocab_size: int, embedding_dim: int):

#     # # Create a random embedding matrix
#     embeddings = nn.Embedding(num_embeddings=vocab_size,
#                               embedding_dim=embedding_dim)
#     # embedding_matrix = np.random.rand(vocab_size, embedding_dim).tolist()
#     # return embedding_matrix
#     pass  # Placeholder for actual implementation


def build_sparse_vector(instance, vocab):
    return torch.tensor([1 if word in instance else 0 for word in vocab], dtype=torch.float32)


if __name__ == "__main__":
    # for quick testing of this module
    dev = PDTBDataset(Path("pdtb/dev.json"))
    print(len(dev))
    print(dev[0])
    # print(
    #     *[
    #         dev[0]["Arg1"]["RawText"],
    #         dev[0]["Arg2"]["RawText"],
    #         dev[0]["Connective"]["RawText"],
    #         dev[0]["Sense"][0],
    #     ],
    #     sep="\n",
    # )
    print(dev.relations[0])
    # print(dev.word2idx)
    # print(dev.idx2word)
    #
    # TODO: add embedding tests
    # TODO: get embeddings working

    print(dev.labels)
    print(dev.label2idx)
    print(dev.idx2label)

    print(dev.relation_indices[0])

    print(dev.__getitem__(0))
    embed, label = dev.__getitem__(0)
    print(embed.shape, label)
