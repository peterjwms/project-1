"""
The model module contains only neural models which are used to be trained.

Instructions:
---
The only implementation for this module is implementing multinomial logistic regression
using the subclass of ``torch.nn.Module``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_embedding_layer(
        embedding_type: str,
        vocab_size: int,
        embedding_dim: int,
        pretrained_embeddings=None
):
    """Initializes an embedding layer based on the specified type"""
    if embedding_type == "glove" and pretrained_embeddings is not None:
        print('Using pretrained embeddings')
        embedding_layer = nn.Embedding.from_pretrained(
            embeddings=pretrained_embeddings, freeze=True)
    elif embedding_type == "random":
        print('Using random embeddings')
        embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim)
    else:
        raise ValueError(
            f"Unsupported embedding type: {embedding_type}")
    return embedding_layer


def aggregate_embeddings(
        embeddings: torch.Tensor,
        method: str = "residual",
) -> torch.Tensor:
    """Aggregates embeddings using the specified method"""
    if method == "average":
        return embeddings.mean(dim=1)
    elif method == "sum":
        return embeddings.sum(dim=1)
    elif method == "residual":
        avg_embeddings = embeddings.mean(dim=1)
        conn_embedding = embeddings[:, 0, :]
        return avg_embeddings + conn_embedding
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")


class LogisticRegression(nn.Module):
    """Logistic regression model"""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            output_dim: int,
            embedding_type: str,
            pretrained_embeddings=None,
            aggregation_method: str = "residual"
    ):
        super(LogisticRegression, self).__init__()
        self.aggregation_method = aggregation_method

        self.embedding = initialize_embedding_layer(
            embedding_type, vocab_size, embedding_dim, pretrained_embeddings)

        self.linear = nn.Linear(embedding_dim, output_dim)
        # softmax not needed here b/c doing cross entropy loss which applies softmax internally
        # nn.Softmax(dim=1)  # need to check if dim is correct
        # self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # out = self.linear(x)
        # y_pred = torch.softmax(out, dim=1)
        embeddings = self.embedding(x)
        aggregated_embeddings = aggregate_embeddings(
            embeddings, self.aggregation_method)
        return self.linear(aggregated_embeddings)


class MLP(nn.Module):
    """Multilayer perceptron"""

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_hidden: int,
            embedding_type: str,
            pretrained_embeddings=None,
            aggregation_method: str = "residual"
    ):
        super(MLP, self).__init__()
        self.aggregation_method = aggregation_method

        # prepare embedding layer, either pretrained or random
        self.embedding = initialize_embedding_layer(
            embedding_type, vocab_size, embedding_dim, pretrained_embeddings)

        self.num_hidden = num_hidden
        if self.num_hidden == 0:
            self.model = nn.Linear(embedding_dim, output_dim)
        elif self.num_hidden == 1:
            self.model = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            # build hidden layers dynamically
            layers = nn.ModuleList()
            # input layer
            layers.append(nn.Linear(embedding_dim, hidden_dim))
            layers.append(nn.ReLU())
            # requested # hidden layers
            for _ in range(num_hidden - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        embeddings = self.embedding(x)
        # print(embeddings.shape)
        aggregated_embeddings = aggregate_embeddings(
            embeddings, self.aggregation_method)
        # print(aggregated_embeddings.shape)
        return self.model(aggregated_embeddings)
        # return self.model(combined_embedding)
        # return self.model(x)


class CNN(nn.Module):
    """CNN model"""

    # TODO: implement CNN model for text classification

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        embedding_type: str,
        num_hidden: int,
        kernel_size: int,
        output_dim: int,
        pretrained_embeddings=None,
    ):
        super(CNN, self).__init__()
        self.embedding = initialize_embedding_layer(
            embedding_type, vocab_size, embedding_dim, pretrained_embeddings)
        self.conv1d = nn.Conv1d(
            in_channels=embedding_dim, out_channels=num_hidden, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        # self.pooling = nn.MaxPool1d()
        self.linear = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        # x = x.unsqueeze(1)  # Add channel dimension
        # print(x)
        # print(x.shape)
        embeddings = self.embedding(x)
        # (batch_size, input_dim, seq_len)
        embeddings = torch.transpose(embeddings, 1, 2)
        # print(embeddings.shape)
        conv_out = self.conv1d(embeddings)
        # print(conv_out.shape)
        activated = self.relu(conv_out)
        # print(activated.shape)
        pooled = F.max_pool1d(activated, activated.size(-1))
        # print(pooled.shape)
        flattened = torch.squeeze(pooled, -1)
        # print(f'{flattened.shape=}')
        logits = self.linear(flattened)
        # print(logits.shape)
        return logits
