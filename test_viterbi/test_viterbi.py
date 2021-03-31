# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math
import numpy as np

def main():
    test_sentences, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["START"]
    prediction = []
    
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""
    states = tuple(emission.keys())

    for test in test_sentences:

        viterbi = np.zeros((len(states), len(test)))
        backpointer = np.zeros((len(states), len(test)))

        for s in range(len(states)):
            viterbi[s][0] = initial[states[s]] * emission[states[s]][test[0]]

        for t in range(1, len(test)):
            for s in range(len(states)):

                maxValue = 0
                maxValueIndex = 0
                for state in range(len(states)):
                    calc = viterbi[state][t-1] * transition[states[state]][states[s]] * emission[states[s]][test[t]]
                    if calc > maxValue:
                        maxValue = calc
                        maxValueIndex = state

                viterbi[s][t] = maxValue
                backpointer[s][t] = maxValueIndex

        bestPathPointer = 0
        bestPathProb = 0
        for s in range(len(states)):
            if viterbi[s][len(test) - 1] > bestPathProb:
                bestPathPointer = s 
                bestPathProb = viterbi[s][len(test) - 1]

        bestPath = [0] * len(test)
        bestPath[len(test) - 1] = states[bestPathPointer]
        for t in range(len(test) - 2, -1, -1):
            bestPath[t] = states[int(backpointer[bestPathPointer, t + 1])]

    for t in range(len(bestPath)):
        bestPath[t] = (test[t], bestPath[t])

    print('Your Output is:',bestPath,'\n Expected Output is:',output)


if __name__=="__main__":
    main()