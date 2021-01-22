#!/usr/bin/env python3
"""Function that performs semantic search on a corpus of documents"""

import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Function that performs semantic search on a corpus of documents"""
    refs = [sentence]
    for file in os.listdir(corpus_path):
        if ".md" in file:
            name = corpus_path + "/" + file
            with open(name, 'r', encoding='utf-8') as f:
                refs.append(f.read())
    model = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    embed = hub.load(model)
    embeddings = embed(refs)
    correlation = np.inner(embeddings, embeddings)
    idx = np.argmax(correlation[0, 1:]) + 1

    return refs[idx]
