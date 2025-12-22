"""
The model module contains only neural models which are used to be trained.

Instructions:
---
The only implementation for this module is implementing multinomial logistic regression
using the subclass of ``torch.nn.Module``.
"""

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Logistic regression model"""

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.Softmax(dim=1)  # need to check if dim is correct
        )

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    """Multilayer perceptron"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_type, pretrained_embeddings=None):
        super(MLP, self).__init__()

        # TODO: undo this and go back to the previous model architecture w/o embeddings as a layer here
        # if embedding_type == "glove":
        #     self.embedding = nn.Embedding.from_pretrained(
        #         embeddings=pretrained_embeddings, freeze=False)
        # elif embedding_type == "random":
        #     self.embedding = nn.Embedding(
        #         num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),

        )

    def forward(self, x):
        # Average embeddings of words in the instance
        # avg_embeddings = self.embedding(x).mean(dim=1)
        # # add connective embedding again to give it more weight
        # conn_embedding = self.embedding(x[:, 0])
        # combined_embedding = avg_embeddings + conn_embedding
        # return self.model(combined_embedding)
        return self.model(x)


class CNN(nn.Module):
    """CNN model"""

    # TODO: implement CNN model for text classification

    def __init__(self, input_dim, num_filters, filter_sizes, output_dim):
        super(CNN, self).__init__()
        # self.embedding = nn.Embedding(input_dim, embed_dim)
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=num_filters, kernel_size=filter_sizes[0])
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=2)
        self.linear = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        # self.softmax = nn.Softmax(dim=1)  # need to check if dim is correct

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        print(x.shape)
        conv_out = self.conv1d(x)
        print(conv_out.shape)
        activated = self.relu(conv_out)
        print(activated.shape)
        pooled = self.pooling(activated, )
        print(pooled.shape)
        flattened = pooled.squeeze(-1)
        print(f'{flattened.shape=}')
        logits = self.linear(flattened)
        print(logits.shape)
        return logits
