from utils import *

word_sim = simple_word_similarity('boat', 'ship')
print('Example of word similarity: ', word_sim)

sentence_sim = simple_sentence_similarity('float boat with a high human', 'a dog in a big ship')
print('Example of sentence similarity: ', sentence_sim)

with open('mc.csv', newline='') as csvfile:
    mc_28 = list(csv.reader(csvfile,delimiter=';'))

mc_28 = np.array(mc_28)

#list of calculated similarities
#Optimal condition is max_num = 100
sim_cal = np.array(words_similarity_dataset(mc_28,max_num = 100))#Change max number of results from DatamuseAPI from here
#list of reference similarities
sim_ref = mc_28[:,2]
sim_ref = sim_ref.astype(float)
corr = pearson_correlation(sim_cal,sim_ref)
