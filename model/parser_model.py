#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parser model for document parsing."""

import os
from model.neural_network import NeuralNetwork
from torch import optim, from_numpy, argmax, save, load
from torch.nn import CrossEntropyLoss
from datetime import datetime
from time import strftime
import numpy as np
import math


class ParserModel():
    """Parser model to be trained for document parsing."""

    CLASSES = {"shift": 0, "arc": 1}

    def __init__(self, n_features, n_buffer_items=3, n_stack_items=3,
                 n_linked_stack_items=2, n_linked_items=2,
                 n_linked_to_linked_items=1, hidden_size=200,
                 dropout_prob=0.5, learning_rate=0.0005, batch_size=1024,
                 model_folder="data/models"):
        """Initialize the parser model.

        Args:
            n_features (int): Number of input features
            hidden_size (int): Hidden layer size
            dropout_prob (float): Dropout probability

        Attributes:
            n_features (int): Number of input features
            hidden_size (int): Hidden layer size
            dropout_prob (float): Dropout probability
            input_to_hidden (Layer): Hidden layer
            hidden_to_logits (Layer): Output layer
            dropout (Layer): Dropout layer
        """
        self.null = -1
        self.root = -2
        self.n_features = n_features
        self.n_buffer_items = n_buffer_items
        self.n_stack_items = n_stack_items
        self.n_linked_stack_items = n_linked_stack_items
        self.n_linked_items = n_linked_items
        self.n_linked_to_linked_items = n_linked_to_linked_items
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_folder = model_folder
        self.name = ""

        network_input_size = n_features * (
                             n_buffer_items + n_stack_items +
                             n_linked_stack_items *
                             (n_linked_items + n_linked_to_linked_items)
                             )

        self.nn = NeuralNetwork(network_input_size, hidden_size, 2,
                                dropout_prob)
        self.optimizer = optim.Adam(self.nn.parameters(),
                                    lr=self.learning_rate)
        self.loss_function = CrossEntropyLoss()

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

    def train(self, train_dataset, dev_dataset, n_epochs=10):
        """Train parser model for n_epochs."""
        self._set_training_session_name()

        X_train, Y_train = self._format_dataset_for_parsing(train_dataset)
        X_dev, Y_dev = self._format_dataset_for_parsing(dev_dataset)

        best_dev_loss = 1e10
        for epoch in range(n_epochs):
            print("Epoch {}/{}...".format(epoch + 1, n_epochs))
            dev_loss = self.train_for_epoch(X_train, Y_train, X_dev, Y_dev)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                self.save_model(self.name)
                print("New best dev loss!")
            print("")

    def train_for_epoch(self, X_train, Y_train, X_dev, Y_dev):
        """Train parser model on one epoch."""
        self.nn.train()
        minibatches = [(X_train[x:x+self.batch_size],
                        Y_train[x:x+self.batch_size])
                       for x in range(0, len(X_train)-self.batch_size,
                                      self.batch_size)]

        for (X, Y) in minibatches:
            self.optimizer.zero_grad()
            X = from_numpy(np.array(X)).long()
            Y = from_numpy(np.array(Y)).long()
            logits = self.nn(X)
            loss = self.loss_function(logits, Y)
            loss.backward()
            self.optimizer.step()

        print("Evaluating on dev set",)
        self.nn.eval()
        X = from_numpy(np.array(X_dev)).long()
        Y = from_numpy(np.array(Y_dev)).long()
        logits = self.nn(X)
        dev_loss = self.loss_function(logits, Y)
        print("- dev loss: {:.2f}".format(dev_loss))
        return dev_loss

    def predict(self, dataset):
        """Predict dependencies for dataset."""
        Y_list = []
        for document in dataset.documents:
            Y = self._predict_dependencies(document.X)
            Y_list.append(Y)
        return Y_list

    def save_model(self, name):
        """Save neural network state."""
        info_filename = "{}/{}.info".format(self.model_folder, name)
        with open(info_filename, "w") as f:
            attrs = vars(self)
            f.write('\n'.join("%s: %s" % item for item in attrs.items()))
            
        filename = "{}/{}.weights".format(self.model_folder, name)
        save(self.nn.state_dict(), filename)
        return filename

    def load_model(self, name):
        """Load neural network state from file."""
        filename = "{}/{}.weights".format(self.model_folder, name)
        self.nn.load_state_dict(load(filename))

    def _set_training_session_name(self):
        """Get name for current training session."""
        self.name = datetime.now().strftime("%Y%m%d-%H%M")

    def _format_dataset_for_parsing(self, dataset):
        """Extract X and Y of dataset documents and convert to transition."""
        X_transition_list = []
        Y_transition_list = []
        for document in dataset.documents:
            X_transition, Y_transition = self._convert_dependencies_to_transition_problem(
                                            document.X, document.Y)
            X_transition_list += X_transition
            Y_transition_list += Y_transition
        return X_transition_list, Y_transition_list

    def _convert_dependencies_to_transition_problem(self, X, Y):
        """Convert dependencies problem to stack manipulation problem."""
        X_transition_list = []
        Y_transition_list = []

        buffer = self._generate_buffer_with_grountruth(X, Y)
        stack = []
        links = []
        while len(buffer) > 0 or len(stack) > 0:
            X_transition = self._generate_X_transition(X, stack, buffer, links)

            # If empty stack -> shift
            if len(stack) == 0:
                Y_transition = self.CLASSES["shift"]
                stack, buffer, links = self._shift(stack, buffer, links)

            else:
                # Look up first stack item
                item = stack[-1]
                item_index = item["index"]

                # Check if has upcoming links
                has_upcoming_links = [upcoming_item["Y"] == item_index
                                      for upcoming_item in buffer]
                has_upcoming_links = True in has_upcoming_links

                # If upcoming links -> shift
                if has_upcoming_links:
                    Y_transition = self.CLASSES["shift"]
                    stack, buffer, links = self._shift(stack, buffer, links)

                else:
                    # Element linked to rook -> arc
                    if item["Y"] == -1:
                        Y_transition = self.CLASSES["arc"]
                        arc = (item["index"], -1)
                        stack, buffer, links = self._arc(arc, stack, buffer,
                                                         links)

                    # Element linked to previous element in stack -> arc
                    elif len(stack) > 1 and item["Y"] == stack[-2]["index"]:
                        Y_transition = self.CLASSES["arc"]
                        arc = (item["index"], item["Y"])
                        stack, buffer, links = self._arc(arc, stack, buffer,
                                                         links)

                    else:
                        raise RuntimeError("Houston, we have a problem")

            X_transition_list.append(X_transition)
            Y_transition_list.append(Y_transition)

        return X_transition_list, Y_transition_list

    def _convert_transitions_to_dependencies(self, X, Y_transition):
        """Generate dependencies from successive transitions."""
        Y_transition.reverse()

        buffer = self._generate_buffer(X)
        stack = []
        links = []
        while len(Y_transition) > 0:
            transition = Y_transition.pop()

            if transition == self.CLASSES["shift"]:
                stack, buffer, links = self._shift(stack, buffer, links)

            elif transition == self.CLASSES["arc"]:
                if len(stack) > 1:
                    arc = (stack[-1]["index"], stack[-2]["index"])
                else:
                    arc = (stack[-1]["index"], -1)
                stack, buffer, links = self._arc(arc, stack, buffer, links)

            else:
                raise TypeError("Invalid transitions")

        Y = [-2 for index in range(len(X))]
        for link in links:
            Y[link[0]] = link[1]

        if -2 in Y:
            raise RuntimeError("Invalid dependencies generation")

        return Y

    def _predict_dependencies(self, X):
        """Generate dependencies an input X."""
        self.nn.eval()
        buffer = self._generate_buffer(X)
        stack = []
        links = []
        while len(buffer) > 0 or len(stack) > 0:
            X_transition = self._generate_X_transition(X, stack, buffer, links)
            X_transition = from_numpy(np.array(X_transition)).long()

            # If buffer empty, arc is only option
            if len(buffer) == 0:
                transition = self.CLASSES["arc"]
            # If stack empty, stack is only option
            elif len(stack) == 0:
                transition = self.CLASSES["shift"]
            else:
                logits = self.nn(X_transition)
                transition = argmax(logits, dim=0)

            if transition == self.CLASSES["shift"]:
                stack, buffer, links = self._shift(stack, buffer, links)

            elif transition == self.CLASSES["arc"]:
                if len(stack) > 1:
                    arc = (stack[-1]["index"], stack[-2]["index"])
                else:
                    arc = (stack[-1]["index"], -1)
                stack, buffer, links = self._arc(arc, stack, buffer, links)

            else:
                raise TypeError("Invalid transitions")

        Y = [-2 for index in range(len(X))]
        for link in links:
            Y[link[0]] = link[1]

        if -2 in Y:
            raise RuntimeError("Invalid dependencies generation")

        return Y

    def _shift(self, stack, buffer, links):
        """Update stack, buffer and links according to shift transition."""
        n_stack = list(stack)
        n_buffer = list(buffer)
        item = n_buffer.pop()
        n_stack.append(item)
        return n_stack, n_buffer, links

    def _arc(self, arc, stack, buffer, links):
        """Update stack, buffer and links according to arc transition."""
        n_stack = list(stack)
        n_links = list(links)
        n_stack.pop()
        n_links.append(arc)
        return n_stack, buffer, n_links

    def _generate_buffer_with_grountruth(self, X, Y):
        """Create buffer from list of items with groundtruth."""
        buffer = [{"index": index, "X": X[index], "Y": Y[index]}
                  for index in range(len(X))]
        buffer.reverse()
        return buffer

    def _generate_buffer(self, X):
        """Create buffer from list of items."""
        buffer = [{"index": index, "X": X[index]} for index in range(len(X))]
        buffer.reverse()
        return buffer

    def _generate_X_transition(self, X, i_stack, i_buffer, i_links):
        """Create features for stack manipulation problem."""
        stack = list(i_stack)
        buffer = list(i_buffer)
        links = list(i_links)

        # Get first stack elements
        stack = self._complete_array(stack, self.n_stack_items,
                                     {"index": None,
                                      "X": [self.null] * self.n_features})
        X_new = [feature
                 for item in stack[:self.n_stack_items]
                 for feature in item["X"]]

        # Get first buffer elements
        buffer = self._complete_array(buffer, self.n_buffer_items,
                                      {"index": None,
                                       "X": [self.null] * self.n_features})
        X_new += [feature
                  for item in buffer[:self.n_buffer_items]
                  for feature in item["X"]]

        # Get elements linked to first stack elements
        for item in stack[-self.n_linked_stack_items:]:
            linked_items_indices = self._get_linked_items(item["index"], links)
            linked_items = [{"index": index, "X": X[index]}
                            for index in linked_items_indices]
            linked_items = self._complete_array(
                            linked_items,
                            self.n_linked_items,
                            {"index": None,
                             "X": [self.null] * self.n_features})
            X_new += [feature
                      for item in linked_items[:self.n_linked_items]
                      for feature in item["X"]]

            # Investigate more the first linked item
            linked_item = linked_items[0]
            linked_to_linked_items_indices = self._get_linked_items(
                                             linked_item["index"], links)
            linked_to_linked_items = [{"index": index, "X": X[index]}
                                      for index in
                                      linked_to_linked_items_indices]
            linked_to_linked_items = self._complete_array(
                                      linked_to_linked_items,
                                      self.n_linked_to_linked_items,
                                      {"index": None,
                                       "X": [self.null] * self.n_features})
            added_items = linked_to_linked_items[
                            -self.n_linked_to_linked_items:]
            X_new += [feature
                      for item in added_items
                      for feature in item["X"]]

        return X_new

    def _complete_array(self, array, target_length, item):
        """Complete array with item to reach given length."""
        while(len(array) < target_length):
            array.insert(0, item)
        return array

    def _get_linked_items(self, index, links):
        """Search for items linked to index in links list."""
        if index is None:
            return []

        return sorted([link[0] for link in links if link[1] == index])
