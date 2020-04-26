#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Document manipulation module."""


class Document():
    """Document features and parsing groundtruth."""

    def __init__(self, X, Y, text):
        """Initialize the document."""
        self.X = X
        self.Y = Y
        self.text = text
