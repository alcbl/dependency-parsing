#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train et evaluate a dependency model."""


from input.dataset import Dataset
from model.parser_model import ParserModel

path = "/Users/alicia/Documents/01-Projets/04-Research/01-Dependency-parsing"

train_dataset = Dataset.create_dataset_from_files(path + "/data/train.feat",
                                                  path + "/data/train.text")
dev_dataset = Dataset.create_dataset_from_files(path + "/data/dev.feat",
                                                path + "/data/dev.text")
test_dataset = Dataset.create_dataset_from_files(path + "/data/test.feat",
                                                 path + "/data/test.text")

n_features = len(train_dataset.documents[0].X[0])

parser_model = ParserModel(n_features, dropout_prob=0.6, learning_rate=0.00001,
                           batch_size=5, hidden_size=100, model_folder=path+"/data/models")
dev_loss = parser_model.train(train_dataset, dev_dataset, 400)

Y = parser_model.predict(test_dataset)
for index, document in enumerate(test_dataset.documents):
    print("Accuracy: " + document.compute_accuracy(Y[index]))
