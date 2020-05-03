#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for `parser_model` module."""

import os
from torch import eq, all
from unittest import TestCase
from model.parser_model import ParserModel


class TestParserModel(TestCase):
    """ParserModel class unit tests."""

    X = [[3, 1], [2.4, 4], [1, -4], [2, 1], [3, 3]]
    Y = [-1, -1, 1, 2, 2]
    links = [(0, -1), (3, 2), (2, 1), (1, -1)]

    def test_init(self):
        """Test model initialization without arguments."""
        model = ParserModel(1)
        self.assertEqual(model.nn.n_features, 12)
        self.assertEqual(model.nn.n_classes, 2)

    def test_generate_buffer_with_groundtruth(self):
        """Test buffer with groundtruth generation."""
        model = ParserModel(2)
        buffer = model._generate_buffer_with_grountruth(self.X, self.Y)
        self.assertEqual(buffer.pop(), {"X": [3, 1], "Y": -1, "index": 0})

    def test_get_linked_items(self):
        """Test simple linked items search."""
        model = ParserModel(2)
        linked_items = model._get_linked_items(2, self.links)
        self.assertEqual(linked_items, [3])

    def test_get_linked_items_for_none(self):
        """Test linked items search with index None."""
        model = ParserModel(2)
        linked_items = model._get_linked_items(None, self.links)
        self.assertEqual(linked_items, [])

    def test_complete_array(self):
        """Test complete array too small."""
        array = [1, 2, 3]
        model = ParserModel(2)
        array = model._complete_array(array, 5, -1)
        self.assertEqual(array, [-1, -1, 1, 2, 3])

    def test_complete_big_array(self):
        """Test complete array already ok."""
        array = [1, 2, 3, 4, 5]
        model = ParserModel(2)
        array = model._complete_array(array, 5, -1)
        self.assertEqual(array, [1, 2, 3, 4, 5])

    def test_generate_transition_empty(self):
        """Test generate transition with empty stack and buffer."""
        model = ParserModel(2)
        X = model._generate_X_transition(self.X, [], [], [])
        self.assertEqual(len(X), model.nn.n_features)

    def test_generate_transition_2(self):
        """Test generate transition."""
        model = ParserModel(2)
        stack = [{"index": 1, "X": self.X[1]},
                 {"index": 2, "X": self.X[2]}]
        buffer = [{"index": 4, "X": self.X[4]}]
        links = [(0, -1), (3, 2)]
        X = model._generate_X_transition(self.X, stack, buffer, links)
        self.assertEqual(len(X), model.nn.n_features)

    def test_shift(self):
        """Test shift transition."""
        model = ParserModel(2)
        stack = [{"index": 1, "X": self.X[1], "Y": -1},
                 {"index": 2, "X": self.X[2], "Y": 1}]
        buffer = [{"index": 4, "X": self.X[4], "Y": 2}]
        links = [(0, -1), (3, 2)]
        n_stack, n_buffer, n_links = model._shift(stack, buffer, links)
        self.assertEqual(n_stack, stack + [buffer[0]])
        self.assertEqual(n_buffer, [])
        self.assertEqual(n_links, links)

    def test_arc(self):
        """Test arc transition."""
        model = ParserModel(2)
        stack = [{"index": 1, "X": self.X[1], "Y": -1},
                 {"index": 2, "X": self.X[2], "Y": 1}]
        buffer = [{"index": 4, "X": self.X[4], "Y": 2}]
        links = [(0, -1), (3, 2)]
        n_stack, n_buffer, n_links = model._arc((2, 1), stack, buffer, links)
        self.assertEqual(n_stack, stack[:-1])
        self.assertEqual(n_buffer, buffer)
        self.assertEqual(n_links, links + [(2, 1)])

    def test_conversions(self):
        """Test transition to dependencies conversions."""
        model = ParserModel(2)
        X_t, Y_t = model._convert_dependencies_to_transition_problem(self.X,
                                                                     self.Y)
        Y_n = model._convert_transitions_to_dependencies(self.X, Y_t)
        self.assertEqual(self.Y, Y_n)

    def test_save_model(self):
        """Test saving model."""
        model = ParserModel(2)
        filename = model.save_model("test")
        self.assertTrue(os.path.exists(filename))

    def test_load_model(self):
        """Test loading model."""
        model = ParserModel(2)
        model.save_model("test")

        model_new = ParserModel(2)
        model_new.load_model("test")
        for layer in model.nn.state_dict().keys():
            self.assertTrue(all(eq(model.nn.state_dict()[layer],
                                model_new.nn.state_dict()[layer])))
