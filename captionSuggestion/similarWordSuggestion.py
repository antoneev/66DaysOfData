import numpy as np
from scipy import spatial
import objectDetection

embeddings_dict = {}
maxReturnWords = 0

with open("files/glove.6B.300d.txt", 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        token = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[token] = vector

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

def main():
    print('Finding Similar Word...')
    for i in range(len(objectDetection.ListofObjects)):
        try:
            print(find_closest_embeddings(embeddings_dict[objectDetection.ListofObjects[i]])[1:maxReturnWords])
        except:
            continue
    print('Similar Word Found...')

if __name__ == '__main__':
    main()
