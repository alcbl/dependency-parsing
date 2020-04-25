#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for `neural_network` module."""

from unittest import TestCase
from model.neural_network import NeuralNetwork
import numpy as np
from torch import tensor


class TestNeuralNetwork(TestCase):
    """NeuratNetwork class unit tests."""

    def test_init(self):
        """Test network initialization without arguments."""
        nn = NeuralNetwork()
        self.assertEqual(nn.n_features, nn.input_to_hidden.in_features)
        self.assertEqual(nn.hidden_size, nn.input_to_hidden.out_features)
        self.assertEqual(nn.hidden_size, nn.hidden_to_logits.in_features)
        self.assertEqual(nn.n_classes, nn.hidden_to_logits.out_features)

    def test_init_with_arguments(self):
        """Test network initialization with arguments."""
        nn = NeuralNetwork(100, 400, 10, 0.2)
        self.assertEqual(nn.input_to_hidden.in_features, 100)
        self.assertEqual(nn.input_to_hidden.out_features, 400)
        self.assertEqual(nn.hidden_to_logits.in_features, 400)
        self.assertEqual(nn.hidden_to_logits.out_features, 10)
        self.assertEqual(nn.dropout.p, 0.2)

    def test_forward(self):
        """Test network forward step."""
        nn = NeuralNetwork()
        input = tensor(np.random.rand(30, nn.n_features))
        output = nn(input)
        output_size = list(output.size())
        self.assertEqual(output_size[0], 30)
        self.assertEqual(output_size[1], nn.n_classes)
