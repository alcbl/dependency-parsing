#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset manipulation module."""

from input.document import Document


class Dataset():
    """Groups documents into a dataset."""

    def __init__(self, documents):
        """Initialize from list of documents."""
        self.documents = documents

    @staticmethod
    def create_dataset_from_files(feature_file, text_file=None):
        """Create a dataset from feature and text files."""
        text_dict = {}
        if text_file:
            with open(text_file, "r") as f:
                for line in f.readlines():
                    elements = line.split("\t")
                    if len(elements) < 3:
                        continue
                    key = "-".join(elements[:2])
                    text_dict[key] = elements[2]

        documents = []
        X_document, Y_document, text_document = [], [], []
        with open(feature_file, "r") as f:
            for line in f.readlines():
                elements = line.split("\t")

                if len(elements) < 3:
                    if len(X_document) > 0:
                        documents.append(Document(X_document, Y_document,
                                                  text_document))
                    X_document, Y_document, text_document = [], [], []
                    continue

                key = "-".join(elements[:2])
                if key in text_dict.keys():
                    text_document.append(text_dict[key])
                else:
                    text_document.append("")

                Y_document.append(int(elements[2]))
                X_document.append([float(elem) for elem in elements[3:]])

            if len(X_document) > 0:
                documents.append(Document(X_document, Y_document,
                                          text_document))

        return Dataset(documents)
