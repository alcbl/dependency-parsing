#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Parser model for document parsing."""

from model.neural_network import NeuralNetwork


class ParserModel():
    """Parser model to be trained for document parsing."""

    CLASSES = {"shift": 0, "arc": 1}

    def __init__(self, n_features, n_buffer_items=3, n_stack_items=3,
                 n_linked_stack_items=2, n_linked_items=2,
                 n_linked_to_linked_items=1, hidden_size=200,
                 dropout_prob=0.5):
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

        network_input_size = n_features * (
                             n_buffer_items + n_stack_items +
                             n_linked_stack_items *
                             (n_linked_items + n_linked_to_linked_items)
                             )

        self.nn = NeuralNetwork(network_input_size, hidden_size, 2,
                                dropout_prob)

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
                    if item["Y"] == -1 and len(stack) == 1:
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
