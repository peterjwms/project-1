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

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from dataset import PDTBDataset, GloveEmbeddingTypes, build_glove_embeddings, process_glove_file
from model import CNN, MLP, LogisticRegression
import pandas as pd

import click
import random
import numpy as np


def train_loop(model, dataloader, optimizer, loss_fn, device):
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


def test_loop(model, dataloader, loss_fn, device):
    model.eval()
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


def train_and_evaluate(
    hyperparams,
    model_type,
    train,
    dev,
    test,
    device,
    glove_embedding_matrix,
):
    # normalize/cast hyperparameters to safe types
    batch_size = int(hyperparams.get("batch_size", 32))
    embedding_dim = int(hyperparams.get("embedding_dim", 50))
    hidden_dim = int(hyperparams.get("hidden_dim", 128))
    num_hidden = int(hyperparams.get("num_hidden", 1))
    num_epochs = int(hyperparams.get("num_epochs", 5))
    learning_rate = float(hyperparams.get("learning_rate", 0.001))
    embedding_type = hyperparams.get("embedding_type", "random")

    if model_type == "logistic_regression":
        model = LogisticRegression(
            vocab_size=len(train.word2idx),
            embedding_dim=embedding_dim,
            output_dim=train.num_classes,
            embedding_type=embedding_type,
            pretrained_embeddings=glove_embedding_matrix
        )
    elif model_type == "mlp":
        model = MLP(
            vocab_size=len(train.word2idx),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=train.num_classes,
            embedding_type=embedding_type,
            num_hidden=num_hidden,
            pretrained_embeddings=glove_embedding_matrix
        )
    elif model_type == "cnn":
        model = CNN(
            vocab_size=len(train.word2idx),
            embedding_dim=embedding_dim,
            embedding_type=embedding_type,
            num_hidden=num_hidden,
            kernel_size=3,
            output_dim=train.num_classes,
            pretrained_embeddings=glove_embedding_matrix
        )
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)

    epochs = num_epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loop(model, train_dataloader, optimizer, loss_fn, device)
        dev_results = test_loop(model, dev_dataloader, loss_fn, device)

    print("Training complete.")
    print("Evaluating on test set...")
    test_results = test_loop(model, test_dataloader, loss_fn, device)
    return test_results


def prep_for_training(max_seq_len=50, embedding_dim=50):
    # set reproducible seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TODO: maybe make this an argument to take
    level = 1

    train = PDTBDataset(data_path=Path("pdtb/train.json"),
                        # embeddings=glove_embeddings,
                        # embedding_dim=hyperparams["embedding_dim"],
                        level=level,
                        max_seq_len=max_seq_len)

    dev = PDTBDataset(data_path=Path("pdtb/dev.json"),
                      #   embeddings=glove_embeddings,
                      #   embedding_dim=hyperparams["embedding_dim"],
                      level=level,
                      max_seq_len=max_seq_len)

    test = PDTBDataset(data_path=Path("pdtb/test.json"),
                       #    embeddings=glove_embeddings,
                       #    embedding_dim=hyperparams["embedding_dim"],
                       level=level,
                       max_seq_len=max_seq_len)

    # build glove embeddings matrix from training vocab
    if embedding_dim == 50:
        glove_path = Path(GloveEmbeddingTypes.glove50.value)
    elif embedding_dim == 100:
        glove_path = Path(GloveEmbeddingTypes.glove100.value)
    elif embedding_dim == 200:
        glove_path = Path(GloveEmbeddingTypes.glove200.value)
    glove_embeddings = process_glove_file(glove_path=glove_path)
    glove_embedding_matrix = build_glove_embeddings(
        vocab=train.idx2word,
        embeddings=glove_embeddings,
        embedding_dim=embedding_dim
    )

    return train, dev, test, glove_embedding_matrix, device


def grid_search():
    hyperparameter_space = {
        'learning_rate': [0.001, 0.005],
        'batch_size': [16, 32, 64],
        'num_epochs': [5, 10, 20],
    }
    # only testing with smaller space here since experiments cover more thorough search
    # embedding_dim = 50
    # embedding_type = "random"
    # hidden_dim = 128
    # num_hidden = 1

    max_seq_len = 50

    train, dev, test, glove_embedding_matrix, device = prep_for_training()

    results = []

    for lr in hyperparameter_space['learning_rate']:
        for batch_size in hyperparameter_space['batch_size']:
            for num_epochs in hyperparameter_space['num_epochs']:
                hyperparams = {
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "embedding_dim": 50,
                    "embedding_type": "random",
                    "max_seq_len": 50,
                    "hidden_dim": 128,
                    "num_hidden": 1,
                }

                test_results = train_and_evaluate(
                    hyperparams,
                    "mlp",
                    train,
                    dev,
                    test,
                    device,
                    glove_embedding_matrix,
                )

                results.append((hyperparams, test_results))

    print("Grid search complete.")
    # print(results)
    results_df = pd.DataFrame(
        results, columns=['Hyperparameters', 'Test Results'])
    # print(results_df)
    hp_df = pd.json_normalize(results_df['Hyperparameters'])
    res_df = pd.json_normalize(results_df['Test Results'])
    results_df = pd.concat(
        [hp_df, res_df], axis=1)

    # save results to a csv file
    # also save the best hyperparameters to a json file that can be loaded later
    results_df.to_csv('experiments/grid_search_results.csv', index=False)

    best_hyperparams = results_df.loc[results_df['accuracy'].idxmax()]
    best_hyperparams.to_json(
        'experiments/grid_search_best_hyperparameters.json')


def experiment_one():
    train, dev, test, glove_embedding_matrix, device = prep_for_training()

    hidden_dims = [64, 128, 256]
    num_hiddens = [1, 2, 4]

    # get this from best hyperparameters from grid search
    with open('experiments/grid_search_best_hyperparameters.json', 'r') as f:
        best_hyperparams = json.load(f)

    hyperparams = {
        "learning_rate": best_hyperparams['learning_rate'],
        "batch_size": best_hyperparams['batch_size'],
        "num_epochs": best_hyperparams['num_epochs'],
        "embedding_type": "random",
        "embedding_dim": 50,
        "max_seq_len": 50,
    }

    results = []
    for hidden_dim in hidden_dims:
        for num_hidden in num_hiddens:
            hyperparams['hidden_dim'] = hidden_dim
            hyperparams['num_hidden'] = num_hidden

            test_results = train_and_evaluate(
                hyperparams,
                "mlp",
                train,
                dev,
                test,
                device,
                glove_embedding_matrix,
            )

            results.append((hidden_dim, num_hidden, test_results))

    results_df = pd.DataFrame(
        results, columns=['hidden_dim', 'num_hidden', 'Test Results'])
    temp_df = pd.json_normalize(results_df['Test Results'])
    results_df = pd.concat(
        [results_df.drop(columns=['Test Results']), temp_df], axis=1)
    results_df.to_csv('experiments/experiment_one_results.csv', index=False)

    best_hyperparams = results_df.loc[results_df['accuracy'].idxmax()]
    best_hyperparams.to_json(
        'experiments/experiment_one_best_hyperparameters.json')


def experiment_two():
    # 2 MLPs for pretrained and random embeddings, with three different sizes
    embed_types = ['glove', 'random']
    embed_dims = [50, 100, 200]

    # get this from best hyperparameters from grid search
    with open('experiments/grid_search_best_hyperparameters.json', 'r') as f:
        best_hyperparams = json.load(f)

    with open('experiments/experiment_one_best_hyperparameters.json', 'r') as f:
        exp_one_hyperparams = json.load(f)

    hyperparams = {
        "learning_rate": float(best_hyperparams['learning_rate']),
        "batch_size": int(best_hyperparams['batch_size']),
        "num_epochs": int(best_hyperparams['num_epochs']),
        "embedding_type": "random",
        "embedding_dim": 50,
        "hidden_dim": int(exp_one_hyperparams['hidden_dim']),
        "max_seq_len": 50,
        "num_hidden": int(exp_one_hyperparams['num_hidden']),
    }

    results = []

    for embed_dim in embed_dims:
        train, dev, test, glove_embedding_matrix, device = prep_for_training(
            embedding_dim=embed_dim)

        for embed_type in embed_types:
            hyperparams['embedding_type'] = embed_type
            hyperparams['embedding_dim'] = int(embed_dim)

            test_results = train_and_evaluate(
                hyperparams,
                "mlp",
                train,
                dev,
                test,
                device,
                glove_embedding_matrix,
            )

            results.append((embed_type, embed_dim, test_results))

    results_df = pd.DataFrame(
        results, columns=['Embedding Type', 'Embedding Dim', 'Test Results'])
    temp_df = pd.json_normalize(results_df['Test Results'])
    results_df = pd.concat(
        [results_df.drop(columns=['Test Results']), temp_df], axis=1)
    results_df.to_csv('experiments/experiment_two_results.csv', index=False)

    best_hyperparams = results_df.loc[results_df['accuracy'].idxmax()]
    best_hyperparams.to_json(
        'experiments/experiment_two_best_hyperparameters.json')


def experiment_three():
    # train each model architecture with best hyperparams and best glove size
    with open('experiments/grid_search_best_hyperparameters.json', 'r') as f:
        best_hyperparams = json.load(f)
    with open('experiments/experiment_one_best_hyperparameters.json', 'r') as f:
        exp_one_hyperparams = json.load(f)
    with open('experiments/experiment_two_best_hyperparameters.json', 'r') as f:
        exp_two_hyperparams = json.load(f)
    # get this from experiment two
    optimal_glove_dim = int(exp_two_hyperparams['Embedding Dim'])

    train, dev, test, glove_embedding_matrix, device = prep_for_training(
        embedding_dim=optimal_glove_dim)
    model_types = ['logistic_regression', 'mlp', 'cnn']

    results = []

    for model_type in model_types:
        hyperparams = {
            "learning_rate": float(best_hyperparams['learning_rate']),
            "batch_size": int(best_hyperparams['batch_size']),
            "num_epochs": int(best_hyperparams['num_epochs']),
            "embedding_type": "glove",
            "embedding_dim": optimal_glove_dim,
            "hidden_dim": int(exp_one_hyperparams.get('hidden_dim', 128)),
            "max_seq_len": 50,
            "num_hidden": 1,  # default, modified below based on model type
        }
        if model_type == 'mlp':
            hyperparams['hidden_dim'] = int(exp_one_hyperparams.get(
                'hidden_dim', 128))  # from experiment one/two
            hyperparams['num_hidden'] = int(exp_one_hyperparams.get(
                'num_hidden', 1))  # from experiment one/two
        elif model_type == 'cnn':
            hyperparams['num_hidden'] = 128

        test_results = train_and_evaluate(
            hyperparams,
            model_type,
            train,
            dev,
            test,
            device,
            glove_embedding_matrix,
        )

        results.append((model_type, test_results))

    results_df = pd.DataFrame(
        results, columns=['Model Type', 'Test Results'])
    temp_df = pd.json_normalize(results_df['Test Results'])
    results_df = pd.concat(
        [results_df.drop(columns=['Test Results']), temp_df], axis=1)
    results_df.to_csv('experiments/experiment_three_results.csv', index=False)


def main_testing():
    # for quick testing of this module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hyperparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 5,
        "embedding_type": "glove",  # or "random"
        # "embedding_type": "random",
        "embedding_dim": 50,
        "hidden_dim": 128,
        "max_seq_len": 50,
        "num_hidden": 1,
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
    # print(glove_embedding_matrix)

    model_type = "logistic_regression"

    if model_type == "logistic_regression":
        model = LogisticRegression(
            vocab_size=len(train.word2idx),
            embedding_dim=hyperparams["embedding_dim"],
            output_dim=train.num_classes,
            embedding_type=embedding_type,
            pretrained_embeddings=glove_embedding_matrix
        )
    elif model_type == "mlp":
        model = MLP(
            vocab_size=len(train.word2idx),
            embedding_dim=hyperparams["embedding_dim"],
            hidden_dim=128,
            output_dim=train.num_classes,
            embedding_type=embedding_type,
            num_hidden=1,
            pretrained_embeddings=glove_embedding_matrix
        )
    elif model_type == "cnn":
        model = CNN(
            vocab_size=len(train.word2idx),
            embedding_dim=hyperparams["embedding_dim"],
            embedding_type=embedding_type,
            num_hidden=16,
            kernel_size=3,
            output_dim=train.num_classes,
            pretrained_embeddings=glove_embedding_matrix
        )

    # print(model)

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
        train_loop(model, train_dataloader, optimizer, loss_fn, device)
        dev_results = test_loop(model, dev_dataloader, loss_fn, device)

    print("Training complete.")
    print("Evaluating on test set...")
    test_results = test_loop(model, test_dataloader, loss_fn, device)
    print(test_results)
    with open(f'{model_type}_test_results.json', 'w') as f:
        json.dump(test_results, f)


# had issues getting click to work properly, commenting out for now to run experiments directly manually in main
# @click.command()
# @click.option('--search_grid', is_flag=True, help='Run hyperparameter grid search experiments.')
# @click.option('--exp_one', is_flag=True, help='Run experiment one.')
# @click.option('--exp_two', is_flag=True, help='Run experiment two.')
# @click.option('--exp_three', is_flag=True, help='Run experiment three.')
# @click.option('--all_experiments', is_flag=True, help='Run all experiments.')
# @click.option

def main(
    # search_grid,
    # exp_one,
    # exp_two,
    # exp_three,
    # all_experiments,
):
    # grid_search()
    # run experiments
    # if search_grid:
    #     grid_search()
    # if all_experiments or exp_one:
    # experiment_one()
    # if all_experiments or exp_two:
    # experiment_two()
    # if all_experiments or exp_three:
    experiment_three()


if __name__ == "__main__":
    main()
