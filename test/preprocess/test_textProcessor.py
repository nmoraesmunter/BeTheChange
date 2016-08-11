from unittest import TestCase
from text_processor import TextProcessor


with open('description.xml', 'r') as file:
    xml_example = file.read()
tp = TextProcessor(xml_example)

class TestTextProcessor(TestCase):

    def test_count_words(self):
        self.assertEqual(tp.count_words(), 8)

    def test_count_words_bold(self):

        self.assertEqual(tp.count_words_bold(), 161)

    def test_count_words_italic(self):
        self.assertEqual(tp.count_words_italic(), 3)

    def test_count_capitalized_words(self):
        self.assertEqual(tp.count_capitalized_words(), 2)

    def test_get_hashtag(self):
        self.assertEqual(tp.get_hashtags(), ["WhatTheFork"])

    def test_get_links(self):
        self.assertEqual(len(tp.get_links()), 8)

    def test_get_link_popularity(self):
        links = tp.get_links()
        self.assertEqual(tp.get_link_popularity(links[0]), 256)

    def test_get_clean_text(self):
        self.assertEqual(tp.get_clean_text(), "hello")
