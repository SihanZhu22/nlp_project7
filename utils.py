import os
import json
import requests
from scipy.stats import pearsonr
import numpy as np
import csv
import pandas as pd


# part 1
def datamuse(query, max_num):
    url = 'https://api.datamuse.com/words?ml=%s&max=%d' % (query, max_num)

    response = requests.get(url)
    results = response.text

    results = json.loads(results)

    words = [a['word'] for a in results]

    return words

# part 2
def jaccard_similarity(a, b):
    overlap = [element for element in a if element in b]
    return 1.0 * len(overlap) / ((len(a) + len(b)) - len(overlap))

def simple_word_similarity(w1, w2, max_num=30):
    assert(isinstance(w1, str)), 'the first input parameter is not a string variable'
    assert(isinstance(w2, str)), 'the second input parameter is not a string variable'

    words_1 = datamuse(w1, max_num=max_num)
    words_2 = datamuse(w2, max_num=max_num)

    return jaccard_similarity(words_1, words_2)

# part 4
def simple_sentence_similarity(s1, s2):
    s1 = s1.strip().split()
    s2 = s2.strip().split()
    overlap_1 = None
    for w in s1:
        words = datamuse(w, 500)
        if not overlap_1:
            overlap_1 = words
        else:
            overlap_1 = [element for element in words if element in overlap_1]
    overlap_1.extend(s1)

    overlap_2 = None
    for w in s2:
        words = datamuse(w, 500)
        if not overlap_2:
            overlap_2 = words
        else:
            overlap_2 = [element for element in words if element in overlap_2]
    overlap_2.extend(s2)
    return jaccard_similarity(overlap_1, overlap_2)

if '__main__' == __name__:
    word_sim = simple_word_similarity('boat', 'ship')
    print('word similarity: ', word_sim)

    sentence_sim = simple_sentence_similarity('float boat with a high human', 'a dog in a big ship')
    print('sentence similarity: ', sentence_sim)

#define the function to calculate all the similarities of one dataset
def words_similarity_dataset(Dataset,max_num=30):
    sim_list = []
    for word_pair in Dataset:
        w1 = word_pair[0]
        w2 = word_pair[1]
        similarity = simple_word_similarity(w1,w2,max_num = max_num)
        sim_list.append(similarity)
    return sim_list

#function to calculate pearson coeffficients
def pearson_correlation(data1, data2):
# calculate Pearson's correlation
    corr, _ = pearsonr(data1, data2)
    print('Pearsons correlation: %.3f' % corr)
