from utils import *

word_sim = simple_word_similarity('boat', 'ship')
print('word similarity: ', word_sim)

sentence_sim = simple_sentence_similarity('float boat with a high human', 'a dog in a big ship')
print('sentence similarity: ', sentence_sim)
