# Template from UIUC CS440


def baseline_training(train):
    
    tagset_occurences = {}
    word_tagsets = {}

    for sentence in train:
        for word in sentence:
            if word[1] in tagset_occurences:
                tagset_occurences[word[1]] += 1
            else:
                tagset_occurences[word[1]] = 1
            if word[0] in word_tagsets:
                word_tagsets[word[0]][0] += 1
                if word[1] in word_tagsets[word[0]][1]:
                    word_tagsets[word[0]][1][word[1]] += 1
                else:
                    word_tagsets[word[0]][1][word[1]] = 1
            else:
                word_tagsets[word[0]] = [1, {word[1]: 1}]

    for word in word_tagsets:
            isTrue = 0
            '''
            if len(word_tagsets[word][1].keys()) > 1:
                print(word_tagsets[word])
                isTrue = 1
            '''
            maxCount = ('', 0)
            for tag in word_tagsets[word][1].keys():
                if word_tagsets[word][1][tag] > maxCount[1]:
                    maxCount = (tag, word_tagsets[word][1][tag])
            word_tagsets[word] = maxCount[0]
            #if isTrue:
            #    print(word_tagsets[word])
        
    maxCount = ('', 0)
    for tag in tagset_occurences:
        if tagset_occurences[tag] > maxCount[1]:
            maxCount = (tag, tagset_occurences[tag])

    return word_tagsets, maxCount[0]
                    

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    wordToTagset, mostCommonTag = baseline_training(train)
    
    toReturn = test
    for i in range(len(test)):
        sentence = test[i]
        for j in range(len(sentence)):
            word = sentence[j]
            if word in wordToTagset:
                toReturn[i][j] = (word, wordToTagset[word])
            else:
                toReturn[i][j] = (word, mostCommonTag)

    return toReturn