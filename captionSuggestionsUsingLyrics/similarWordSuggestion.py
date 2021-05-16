import numpy as np
from scipy import spatial
import objectDetection

# Declare variables
embeddings_dict = {}
maxReturnWords = 0
allSimilarWords = {}

# Opening gloVe File
with open("files/glove.6B.300d.txt", 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        token = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[token] = vector

# Finding closest word to input
def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

def main():
    print('Similar Word Suggestion started...') # Indicating algorithm started

    # Passing in each object 1 by 1 using objectDetection NOT allObjects (I didn't find the need to find similar words to colors)
    for i in range(len(objectDetection.ListofObjects)):
        try:
            similarWords = find_closest_embeddings(embeddings_dict[objectDetection.ListofObjects[i]])[1:maxReturnWords]
            allSimilarWords[objectDetection.ListofObjects[i]] = similarWords
        except:
            continue

    print('Similar Word Suggestion completed...') # Indicating algorithm completed

if __name__ == '__main__':
    main()
