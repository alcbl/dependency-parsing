#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for `dataset` module."""

from unittest import TestCase
from input.dataset import Dataset


class TestDataset(TestCase):
    """Dataset class unit tests."""

    TEST_FOLDER = "./test/test_file"

    def test_create_from_files(self):
        """Test dataset creation from files."""
        feature_file = "{}/{}".format(self.TEST_FOLDER, "train.feat")
        text_file = "{}/{}".format(self.TEST_FOLDER, "train.text")

        dataset = Dataset.create_dataset_from_files(feature_file, text_file)
        self.assertEqual(len(dataset.documents), 4)
        self.assertEqual(len(dataset.documents[0].X),
                         len(dataset.documents[0].Y))
        self.assertEqual(len(dataset.documents[0].X),
                         len(dataset.documents[0].text))
