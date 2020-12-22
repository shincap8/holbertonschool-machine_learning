#!/usr/bin/env python3
"""Function that calculates the unigram BLEU score for a sentence"""

import numpy as np


def counter(phrase):
    """Return dict with count of words"""
    dict = {}
    for x in phrase:
        if x not in dict:
            dict[x] = phrase.count(x)
    return (dict)


def count_clip(references, sentence):
    """Count clip"""
    res = {}
    ct_sentence = counter(sentence)
    for ref in references:
        ct_ref = counter(ref)
        for k in ct_ref:
            if k in res:
                res[k] = max(ct_ref[k], res[k])
            else:
                res[k] = ct_ref[k]
    count_clip = {k: min(ct_sentence.get(k, 0),
                         res.get(k, 0)) for k in ct_sentence}
    return (count_clip)


def modified_precision(references, sentence):
    """Modified precision"""
    ct_clip = count_clip(references, sentence)
    ct = counter(sentence)
    return sum(ct_clip.values()) / float(max(sum(ct.values()), 1))


def uni_bleu(references, sentence):
    """Function that calculates the unigram BLEU score for a sentence"""
    W = [0.25 for x in range(4)]
    Pn = [modified_precision(references, sentence)
          for ngram, _ in enumerate(W, start=1)]
    c = len(sentence)
    closest_ref_idx = np.argmin([abs(len(x) - c) for x in references])
    r = len(references[closest_ref_idx])
    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - (float(r) / c))
    score = np.sum([(wn * np.log(Pn[i])) if Pn[i] != 0 else 0
                    for i, wn in enumerate(W)])
    BLEU = BP * np.exp(score)
    if BLEU > 0.4:
        return round(BLEU, 7)
    return BLEU
