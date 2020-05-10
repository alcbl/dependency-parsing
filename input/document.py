#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Document manipulation module."""


class Document():
    """Document features and parsing groundtruth."""

    def __init__(self, name, X, Y, text):
        """Initialize the document."""
        self.name = name
        self.X = X
        self.Y = Y
        self.text = text

    def write_gt(self):
        """Write groundtruth file from Y."""
        filename = "{}.gt".format(self.name)
        with open(filename, "w") as f:
            for index in range(len(self.text)):
                f.write("{}\t{}\t{}\n".format(index, self.Y[index],
                                              self.text[index].strip()))
