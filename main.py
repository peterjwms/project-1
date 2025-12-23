"""
The main script for training the model and display the evaluation results.

Instructions:
---
Commonly, the main script contains following functions:
* ``train_loop``
* ``test_loop``
* Evaluation functions
* (Optional) Command line arguments or options
    * If you need explicit control over this script (e.g. learning rate, training size, etc.)
* (Optional) Any functions from ``utils.py`` that helps display results and evaluation

Eventually, this script should be run as
```
uv run main.py <ARGUMENTS> --<OPTIONS>
```

References:
---
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from dataset import PDTBDataset, GloveEmbeddingTypes, build_glove_embeddings, process_glove_file
from model import CNN, MLP, LogisticRegression


def train_loop(model, dataloader, optimizer, loss_fn):
    model.train()
    for i, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(inputs.size())
        # print(labels.size())
        # print(labels)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i + 1) % 100 == 0:
            print(f"Batch {i+1}, Loss: {loss.item()}")


def test_loop(model, dataloader, loss_fn):
    model.eval()
    # TODO: implement evaluation metrics (accuracy, precision, recall, F1, etc.)
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for i, batch in enumerate(dataloader):
            inputs, labels = batch
            # print(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            acc = utils.accuracy(labels, preds)

            if (i + 1) % 10 == 0:
                # print(labels)
                # print(preds)
                print(f"Batch {i+1}, Loss: {loss.item()}, Accuracy: {acc}")

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        # Compute overall accuracy
        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)
        overall_acc = utils.accuracy(all_labels, all_preds)
        precision = utils.precision(all_labels, all_preds)
        recall = utils.recall(all_labels, all_preds)
        f1 = utils.f1_score(all_labels, all_preds)
        print(f"Overall Accuracy: {overall_acc}")
        print(f"Precision: {utils.precision(all_labels, all_preds)}")
        print(f"Recall: {utils.recall(all_labels, all_preds)}")
        print(f"F1 Score: {utils.f1_score(all_labels, all_preds)}")

    return {"accuracy": overall_acc, "precision": precision, "recall": recall, "f1": f1}

# TODO: could make hyperparameters as arguments, and add argparse for command line options
# e.g. learning rate, batch size, number of epochs, training size, model type
# will need to do a gridsearch for hyperparameter tuning, maybe make that another argparse option


def experiment1():
    results_file = Path("results/experiment1.txt")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    hidden_dim_sizes = [32, 64, 128, 256]
    hidden_layers = [0, 1, 2, 3]
    embedding_types = ["glove", "random"]

    with open(results_file, "w") as f:
        f.write("Experiment 1 Results\n")

    pass  # Placeholder for actual implementation


if __name__ == "__main__":
    # for quick testing of this module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 5,
        # "embedding_type": "glove",  # or "random"
        "embedding_type": "random",
        "embedding_dim": 50,
        "hidden_dim": 128,
        "max_seq_len": 50,
    }

    # TODO: set hyperparameters here (based on argparse if implemented),
    # e.g. learning rate, batch size, number of epochs, training size, model
    embedding_type = hyperparams["embedding_type"]
    # TODO: should load glove embeddings here and pass to dataset

    level = 1

    # returns dataset as sequences of token indices, padding to max_seq_len
    train = PDTBDataset(data_path=Path("pdtb/train.json"),
                        # embeddings=glove_embeddings,
                        # embedding_dim=hyperparams["embedding_dim"],
                        level=level,
                        max_seq_len=hyperparams["max_seq_len"])
    train_dataloader = DataLoader(train, batch_size=32, shuffle=True)

    dev = PDTBDataset(data_path=Path("pdtb/dev.json"),
                      #   embeddings=glove_embeddings,
                      #   embedding_dim=hyperparams["embedding_dim"],
                      level=level,
                      max_seq_len=hyperparams["max_seq_len"])
    dev_dataloader = DataLoader(dev, batch_size=32, shuffle=False)

    test = PDTBDataset(data_path=Path("pdtb/test.json"),
                       #    embeddings=glove_embeddings,
                       #    embedding_dim=hyperparams["embedding_dim"],
                       level=level,
                       max_seq_len=hyperparams["max_seq_len"])
    test_dataloader = DataLoader(test, batch_size=32, shuffle=False)

    # build glove embeddings matrix from training vocab
    glove_path = Path(GloveEmbeddingTypes.glove50.value)
    glove_embeddings = process_glove_file(glove_path=glove_path)
    glove_embedding_matrix = build_glove_embeddings(
        vocab=train.idx2word,
        embeddings=glove_embeddings,
        embedding_dim=hyperparams["embedding_dim"]
    )
    # should be (vocab_size, embedding_dim)
    print(glove_embedding_matrix.shape)
    print(glove_embedding_matrix)

    # model = LogisticRegression(
    #     vocab_size=len(train.word2idx),
    #     embedding_dim=hyperparams["embedding_dim"],
    #     output_dim=train.num_classes,
    #     embedding_type=embedding_type,
    #     pretrained_embeddings=glove_embedding_matrix
    # )
    # model = MLP(
    #     vocab_size=len(train.word2idx),
    #     embedding_dim=hyperparams["embedding_dim"],
    #     hidden_dim=128,
    #     output_dim=train.num_classes,
    #     embedding_type=embedding_type,
    #     num_hidden=1,
    #     pretrained_embeddings=glove_embedding_matrix
    # )
    model = CNN(
        vocab_size=len(train.word2idx),
        embedding_dim=hyperparams["embedding_dim"],
        embedding_type=embedding_type,
        num_hidden=16,
        kernel_size=3,
        output_dim=train.num_classes,
        pretrained_embeddings=glove_embedding_matrix
    )

    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparams["learning_rate"])
    # use CrossEntropyLoss for multi-class classification
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)
    # train_dataloader.to(device)
    # dev_dataloader.to(device)

    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loop(model, train_dataloader, optimizer, loss_fn)
        dev_results = test_loop(model, dev_dataloader, loss_fn)

    print("Training complete.")
    print("Evaluating on test set...")
    test_results = test_loop(model, test_dataloader, loss_fn)
