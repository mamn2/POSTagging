# POSTagger

To run the code on the Brown corpus data you need to tell it where the data is and which algorithm to run, either baseline, viterbi_1, or viterbi_2:

`python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm [baseline, viterbi_1, viterbi_2]`

The baseline algorithm considers each words independently without regard to it's surroundings. For unseen words it just uses the most common part of speech (nouns). It gets about 93.9% accuracy.

Viterbi_1 implements the HMM trellis (Viterbi) decoding algorithm. It can be found in the Jurafsky and Martin textbook on NLP. It gets arround a 93% accuracy.

Viterbi_2 is an extension of Viterbi_1 and uses an improved technique for unseen words. It gets about 95.5% accuracy.