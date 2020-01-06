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

with open('STSSa.csv', newline='') as csvfile:
    stss = list(csv.reader(csvfile,delimiter=';'))

stss = np.array(stss)

sim_cal_sen = np.array(sentence_similarity_dataset(stss))
sim_ref_sen = stss[:,2]
sim_ref_sen = sim_ref_sen.astype(float)
for sim in sim_ref_sen:
    sim = sim/4
corr_sen = pearson_correlation(sim_cal_sen,sim_ref_sen)
