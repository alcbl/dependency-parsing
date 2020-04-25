#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neural network module with pytorch."""

import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Simple neural network with one hidden layer and dropout."""

    def __init__(self, n_features=36, hidden_size=200, n_classes=3,
                 dropout_prob=0.5):
        """Initialize the parser model.

        Args:
            n_features (int): Number of input features
            hidden_size (int): Hidden layer size
            n_classes (int): Number of output classes
            dropout_prob (float): Dropout probability

        Attributes:
            n_features (int): Number of input features
            hidden_size (int): Hidden layer size
            n_classes (int): Number of output classes
            dropout_prob (float): Dropout probability
            input_to_hidden (Layer): Hidden layer
            hidden_to_logits (Layer): Output layer
            dropout (Layer): Dropout layer
        """
        super(NeuralNetwork, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size

        self.input_to_hidden = nn.Linear(in_features=self.n_features,
                                         out_features=self.hidden_size)
        self.hidden_to_logits = nn.Linear(in_features=self.hidden_size,
                                          out_features=self.n_classes)
        self.dropout = nn.Dropout(p=self.dropout_prob)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights.

        We use Xavier Uniform Initialization for weight initialization.
        """
        nn.init.xavier_uniform_(self.input_to_hidden.weight, gain=1)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight, gain=1)

    def forward(self, t):
        """Run the model forward.

        Args:
            t (Tensor): input tensor (batch_size, n_features)

        Returns:
            logits (Tensor): tensor of predictions (batch_size, n_classes)
                             without applying softmax
        """
        x = t.float()
        hidden_state = nn.functional.relu(self.input_to_hidden(x))
        dropped_hidden_state = self.dropout(hidden_state)
        return self.hidden_to_logits(dropped_hidden_state)
