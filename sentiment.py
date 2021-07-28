#!/usr/bin/python
import random
from typing import Callable, Dict, List, Tuple, TypeVar
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# binary classification
############################################################

############################################################
# feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    features = {}
    for word in x.split():
        if word not in features:
            features[word]=1
        else:
            features[word]+=1
    return features

############################################################
#  stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The identity function may be used as the featureExtractor function during testing.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight
    def predictor(x):
        phi_x = featureExtractor(x)
        if dotProduct(phi_x,weights)<0:
            return -1
        return 1
    for x,y in trainExamples:
        for feature in featureExtractor(x):
            weights[feature]=0
    for i in range(numEpochs):
        for x,y in trainExamples:
            phi_x = featureExtractor(x)
            if dotProduct(phi_x,weights)*y < 1.0:
                for elem in phi_x:
                    weights[elem]+=eta*y*phi_x[elem]
        print("Iteration number : {} , Training error : {} , validation error : {} ".format(i,evaluatePredictor(trainExamples,predictor),evaluatePredictor(validationExamples,predictor)))
    # END_YOUR_CODE
    return weights


############################################################
#  generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi={}
        for item in weights.keys():
            phi[item]=random.randint(0,5)
        if dotProduct(phi,weights)>=0:
            return (phi,1)
        return (phi, -1)
    return [generateExample() for _ in range(numExamples)]
############################################################
# character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        phi = {}
        without_space_x = ''.join(x.split(' '))
        for i in range(len(without_space_x)-n+1):
            if without_space_x[i:i+n] not in phi:
                phi[without_space_x[i:i+n]]=1
            else:
                phi[without_space_x[i:i+n]]+=1
        return phi
    return extract
############################################################
