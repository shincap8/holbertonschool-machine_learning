#!/usr/bin/env python3
"""Function that creates a bag of words embedding matrix"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Function that creates a bag of words embedding matrix"""
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    return (X.toarray(), vectorizer.get_feature_names())
