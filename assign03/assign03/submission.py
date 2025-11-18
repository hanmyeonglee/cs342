#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

# You may use this seed
SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################


def problem_1a() -> dict[str, int]:
    """
    return a dictionary that contains the following words as keys:
        pretty, good, bad, plot, not, scenery
    """
    return {
        "pretty": 1,
        "good": 0,
        "bad": -1,
        "plot": -1,
        "not": -1,
        "scenery": 0,
    }


############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction


def extractWordFeatures(x: str) -> dict[str, int]:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    return dict(Counter(x.split()))


############################################################
# Problem 2b: stochastic gradient descent


def learnPredictor(
    trainExamples: list[tuple[str, int]], 
    testExamples: list[tuple[str, int]], 
    featureExtractor: callable, 
    numIters: int, 
    eta: float
):
    """
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    """
    weights = collections.defaultdict(int)  # feature => weight

    def sigmoid(n: float) -> float:
        return 1 / (1 + math.exp(-n))
    
    def boolToInt(b: bool) -> int:
        return 1 if b else 0

    featureExtractedTrainExamples = [(featureExtractor(x), y) for x, y in trainExamples]
    for _ in range(numIters):
        for features, y in featureExtractedTrainExamples:
            error = boolToInt(y == 1) - sigmoid(dotProduct(weights, features))
            for f, v in features.items():
                weights[f] += eta * error * v

    return weights


############################################################
# Problem 2c: bigram features


def extractBigramFeatures(x: str) -> dict[str | tuple[str, str], int]:
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    phi = extractWordFeatures(x)
    
    words = ['<s>'] + x.split() + ['</s>']
    bigram_words = []
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        bigram_words.append(bigram)

    phi.update(Counter(bigram_words))

    return phi
