# template from UIUC CS440

import math
import numpy as np 


def counts(train):

    uniqueTags = set()
    uniqueWords = set()
    transitionCounts = {}
    transitionCountTotals = {}
    emissionCounts = {}

    for sentence in train:
        
        prevTag = 'START'
        for curWord, curTag in sentence:

            uniqueTags.add(curTag)
            uniqueWords.add(curWord)
    
            if curTag not in emissionCounts:
                emissionCounts[curTag] = {}
            if curWord in emissionCounts[curTag]:
                emissionCounts[curTag][curWord] += 1
            else:
                emissionCounts[curTag][curWord] = 1

            if prevTag not in transitionCounts:
                transitionCounts[prevTag] = {}
                transitionCountTotals[prevTag] = 0
            if curTag in transitionCounts[prevTag]:
                transitionCounts[prevTag][curTag] += 1
            else:
                transitionCounts[prevTag][curTag] = 1
            transitionCountTotals[prevTag] += 1

            prevTag = curTag

    return uniqueTags, uniqueWords, transitionCounts, transitionCountTotals, emissionCounts

def calcTransProbs(transitionCounts, transitionCountTotals, uniqueTags, smoothingParam):

    transitionProbs = {}
    for prev in uniqueTags:
        if prev not in transitionProbs:
            transitionProbs[prev] = {}
        for cur in uniqueTags:
            if prev in transitionCounts and cur in transitionCounts[prev]:
                transitionProbs[prev][cur] = math.log( (transitionCounts[prev][cur] + smoothingParam) / (transitionCountTotals[prev] + ( len(uniqueTags) + 1 ) * smoothingParam))
            else:
                transitionProbs[prev][cur] = math.log(smoothingParam / ( sum(transitionCountTotals.values()) + ( len(uniqueTags) + 1 ) * smoothingParam))

    transitionProbs['UNSEEN'] = math.log(smoothingParam / (sum(transitionCountTotals.values()) + ( len(uniqueTags) + 1 ) * smoothingParam))

    return transitionProbs
    
def calcEmissionProbs(emissionCounts, uniqueTags, uniqueWords, smoothingParam):

    emissionProbs = {}
    for word in uniqueWords:
        for tag in uniqueTags:

            if tag not in emissionProbs:
                emissionProbs[tag] = {}
            
            if tag in emissionCounts and word in emissionCounts[tag]:
                emissionProbs[tag][word] = math.log( (emissionCounts[tag][word] + smoothingParam) / ( sum(emissionCounts[tag].values()) + ( len(uniqueWords) + 1 ) * smoothingParam))
            else:
                emissionProbs[tag][word] = math.log(smoothingParam / ( sum(emissionCounts[tag].values()) + ( len(uniqueWords) + 1 ) * smoothingParam))

    emissionProbs['UNSEEN'] = math.log(smoothingParam / ( sum(emissionCounts[tag].values()) + ( len(uniqueWords) + 1 ) * smoothingParam))

    return emissionProbs

def posTagging(sentence, tags, transitionProbs, emissionProbs):

    viterbi = np.zeros((len(sentence), len(tags)))
    backpointer = np.zeros((len(sentence), len(tags)))

    initialProbs = transitionProbs['START']
    for i in range(len(tags)):

        tag = tags[i]
        initialProbsTag = None
        emissionsProbsTag = None

        if tag in initialProbs:
            initialProbsTag = initialProbs[tag]
        else:
            initialProbsTag = initialProbs['UNSEEN']

        if tag in emissionProbs and sentence[0] in emissionProbs[tag]:
            emissionsProbsTag = emissionProbs[tag][sentence[0]]
        else:
            emissionsProbsTag = emissionProbs['UNSEEN']

        viterbi[0][i] = initialProbsTag + emissionsProbsTag

    for i in range(1, len(sentence)):
        word = sentence[i]
        for j in range(len(tags)):

            maxVal = None
            argMax = None

            emissionsProbability = None

            if tags[j] in emissionProbs and word in emissionProbs[tags[j]]:
                emissionsProbability = emissionProbs[tags[j]][word]
            else:
                emissionsProbability = emissionProbs['UNSEEN']
                
            for k in range(len(tags)):

                transitionProbability = None

                if tags[k] in transitionProbs and tags[j] in transitionProbs[tags[k]]:
                    transitionProbability = transitionProbs[tags[k]][tags[j]]
                else:
                    transitionProbability = transitionProbs['UNSEEN']

                if maxVal is None or (emissionsProbability + transitionProbability + viterbi[i - 1][k]) > maxVal:
                    maxVal = emissionsProbability + transitionProbability + viterbi[i - 1][k]
                    argMax = k

            viterbi[i][j] = maxVal
            backpointer[i][j] = argMax

    toReturn = []
    lastWord = len(sentence) - 1
    curMaxVal = None
    curArgMax = None
    for i in range(len(tags)):
        if curMaxVal == None or viterbi[lastWord][i] > curMaxVal:
            curMaxVal = viterbi[lastWord][i]
            curArgMax = i
    for i in range(lastWord, -1, -1):
        toReturn = [(sentence[i], tags[curArgMax])] + toReturn
        curArgMax = int(backpointer[i][curArgMax])

    return toReturn


def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    smoothingParam = 0.0003
    uniqueTags, uniqueWords, transitionCounts, transitionCountTotals, emissionCounts = counts(train)
    transitionProbs = calcTransProbs(transitionCounts, transitionCountTotals, uniqueTags, smoothingParam)
    emissionProbs = calcEmissionProbs(emissionCounts, uniqueTags, uniqueWords, smoothingParam)

    return [posTagging(sentence, list(uniqueTags), transitionProbs, emissionProbs) for sentence in test]
