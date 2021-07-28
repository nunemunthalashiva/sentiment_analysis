#!/usr/bin/python3
from util import *

def load(module_name):
    try:
        return __import__(module_name)
    except Exception as e:
        self.fail("Threw exception when importing '%s': %s" % (module_name, e))
        self.fatalError = True
        return None
    except:
        self.fail("Threw exception when importing '%s'" % module_name)
        self.fatalError = True
        return None
sentiment = load('sentiment')
def test3b2():
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = sentiment.extractWordFeatures
    weights = sentiment.learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))
############################################################
if __name__=='__main__':
    test3b2()
