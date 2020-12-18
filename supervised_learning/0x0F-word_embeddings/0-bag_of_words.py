#!/usr/bin/env python3
"""Function that creates a bag of words embedding matrix"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Function that creates a bag of words embedding matrix"""
    words = []
    if vocab is None:
        for i in sentences:
            sen = i.split(" ")
            for s in sen:
                word = re.split(r'\.|\!|\?|\,|\'', s)
                if vocab is None:
                    words.append(word[0].lower())
        aux = set(words)
        vocab = list(aux)
        vocab.sort()
    embeddings = np.zeros((len(sentences), len(vocab)))
    for i in range(len(sentences)):
        sentence = sentences[i].split(" ")
        for s in sentence:
            word = re.split(r'\.|\!|\?|\,|\'', s)
            if word[0].lower() in vocab:
                embeddings[i][vocab.index(word[0].lower())] += 1
    return (embeddings.astype(int), vocab)
