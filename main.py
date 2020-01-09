from utils import *

# part 3
datasets = ['mc', 'rg', 'wordsim']

for data in datasets:
    print('processing dataset: %s' % data)
    with open('datasets/%s.csv' % data, newline='') as csvfile:
        contents = list(csv.reader(csvfile, delimiter=';'))

    #Change max number of results from DatamuseAPI from here
    sim_cal = np.array(words_similarity_dataset(contents, max_num=1000))
    sim_ref = np.array(contents)[:,2].astype(float)
    corr = pearson_correlation(sim_cal,sim_ref)

    with open('word_similarity.txt', 'a') as simfile:
        simfile.write('Using Datamuse methods\n')
        simfile.write('w1; w2; human_sim; method_sim\n\n')
        for i, pair in enumerate(contents):
            simfile.write('%s;%s;%s;%f\n' % (pair[0], pair[1], pair[2], sim_cal[i]))
        simfile.write('\n\n')

    with open('results.txt', 'a') as resfile:
        resfile.write('pearson correlation in dataset [%s] for Datamuse methods is %f\n' % (data, corr))

exit()

# part 5
with open('datasets/stss-131.csv', encoding='utf8', newline='') as csvfile:
    contents = list(csv.reader(csvfile, delimiter=';'))

sim_cal = np.array(sentence_similarity_dataset(contents)).reshape(-1,)

with open('sentence_similarity.txt', 'a') as simfile:
    simfile.write('Using Datamuse methods\n')
    simfile.write('s1; s2; human_sim; method_sim\n\n')
    for i, pair in enumerate(contents):
        simfile.write('%s;%s;%s;%f\n' % (pair[0], pair[1], pair[2], sim_cal[i] * 4))
    simfile.write('\n\n')

sim_ref = np.array(contents)[:,2].astype(float) / 4.0
corr = pearson_correlation(sim_cal,sim_ref)

with open('results.txt', 'a') as resfile:
    resfile.write('pearson correlation in dataset [%s] for Datamuse methods is %f\n' % ('STS-131', corr))

# part 6
from gensim.models import Word2Vec
import nltk
nltk.download('brown')

from nltk.corpus import brown

with open('datasets/stss-131.csv', newline='') as csvfile:
    contents = list(csv.reader(csvfile, delimiter=';'))

model_word2vec = Word2Vec(brown.sents(), min_count=8)
sim_cal = np.array(sentence_similarity_dataset_model(contents, model_word2vec.wv)).reshape(-1,)

with open('sentence_similarity.txt', 'a') as simfile:
    simfile.write('Using Word2Vec embedding\n')
    simfile.write('s1; s2; human_sim; method_sim\n\n')
    for i, pair in enumerate(contents):
        simfile.write('%s;%s;%s;%f\n' % (pair[0], pair[1], pair[2], sim_cal[i] * 4))
    simfile.write('\n\n')

sim_ref = np.array(contents)[:,2].astype(float) / 4.0
corr = pearson_correlation(sim_cal,sim_ref)

with open('results.txt', 'a') as resfile:
    resfile.write('pearson correlation in dataset [%s] for Word2Vec embedding is %f\n' % ('STS-131', corr))


# part 7_glove

with open('datasets/stss-131.csv', newline='') as csvfile:
    contents = list(csv.reader(csvfile, delimiter=';'))

model_glove = {}
with open("glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        model_glove[word] = vector
sim_cal = np.array(sentence_similarity_dataset_model(contents, model_glove)).reshape(-1,)

with open('sentence_similarity.txt', 'a') as simfile:
    simfile.write('Using Glove embedding\n')
    simfile.write('s1; s2; human_sim; method_sim\n\n')
    for i, pair in enumerate(contents):
        simfile.write('%s;%s;%s;%f\n' % (pair[0], pair[1], pair[2], sim_cal[i] * 4))
    simfile.write('\n\n')

sim_ref = np.array(contents)[:,2].astype(float) / 4.0
corr = pearson_correlation(sim_cal,sim_ref)

with open('results.txt', 'a') as resfile:
    resfile.write('pearson correlation in dataset [%s] for Glove embedding is %f\n' % ('STS-131', corr))

# part 7_fasttext
from gensim.models.fasttext import FastText
import nltk
nltk.download('brown')

from nltk.corpus import brown

with open('datasets/stss-131.csv', newline='') as csvfile:
    contents = list(csv.reader(csvfile, delimiter=';'))

model_fasttext = FastText(brown.sents(), min_count=8)
sim_cal = np.array(sentence_similarity_dataset_model(contents, model_fasttext.wv)).reshape(-1,)

with open('sentence_similarity.txt', 'a') as simfile:
    simfile.write('Using FastText embedding\n')
    simfile.write('s1; s2; human_sim; method_sim\n\n')
    for i, pair in enumerate(contents):
        simfile.write('%s;%s;%s;%f\n' % (pair[0], pair[1], pair[2], sim_cal[i] * 4))
    simfile.write('\n\n')

sim_ref = np.array(contents)[:,2].astype(float) / 4.0
corr = pearson_correlation(sim_cal,sim_ref)

with open('results.txt', 'a') as resfile:
    resfile.write('pearson correlation in dataset [%s] for FastText embedding is %f\n' % ('STS-131', corr))

# part 8
with open('datasets/stss-131.csv', newline='') as csvfile:
    contents = list(csv.reader(csvfile, delimiter=';'))

from sematch.semantic.similarity import YagoTypeSimilarity
sim = YagoTypeSimilarity()
sim_cal = np.array(sentence_similarity_dataset_yago(contents, sim)).reshape(-1,)

with open('sentence_similarity.txt', 'a') as simfile:
    simfile.write('Using Yago concepts\n')
    simfile.write('s1; s2; human_sim; method_sim\n\n')
    for i, pair in enumerate(contents):
        simfile.write('%s;%s;%s;%f\n' % (pair[0], pair[1], pair[2], sim_cal[i] * 4))
    simfile.write('\n\n')

sim_ref = np.array(contents)[:,2].astype(float) / 4.0
corr = pearson_correlation(sim_cal,sim_ref)

with open('results.txt', 'a') as resfile:
    resfile.write('pearson correlation in dataset [%s] for Yago concepts is %f\n' % ('STS-131', corr))

print('done')
