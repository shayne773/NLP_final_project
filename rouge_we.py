from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np



#ROUGE-WE

#file from https://github.com/stanfordnlp/GloVe?tab=readme-ov-file
#each word has a vector of numbers in this file
#it is made so that words with similar meanings (king, queen) or pos-tag (run, walk) can have more similar cos sim
with open("embedding/glove.42B.300d.txt", 'r',encoding="utf-8") as file:
    lines = file.readlines()

word_vectors = defaultdict(list)
for l in lines:
    w = l.strip().split(' ')
    word_vectors[w[0]] = [float(num) for num in w[1:]]



#get: lower case generated summary
#ref: lower case reference summary
#word_vectors: model from gensim to get embedding matrix
def rouge_we(get, ref, word_vectors):

    #tokenizing
    ref_tokens = [word for word in word_tokenize(ref) if word in word_vectors.keys()]
    get_tokens = [word for word in word_tokenize(get) if word in word_vectors.keys()]

    #embedding
    #two matrixes
    #each word to a list, so lists of words become matrix
    ref_embeddings = np.array([word_vectors[word] for word in ref_tokens])
    get_embeddings = np.array([word_vectors[word] for word in get_tokens])

    #cosine similarity of ref and hyp matrix
    #this function compares all words in ref with all words in get
    #it return a {ref_word_count*get_word_count} matrix
    sim_matrix = cosine_similarity(ref_embeddings, get_embeddings)

    #presicion: best match score for each word in ref and calculate mean
    #recall: best match score for each word in get and calculate mean
    precision = sim_matrix.max(axis=1).mean()
    recall = sim_matrix.max(axis=0).mean() 

    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, F1


get_summary = "a quick brown fox jumps over the lazy dog"
ref_summary = "the swift fox leaps over a sleeping dog"

result = rouge_we(get_summary.lower(), ref_summary.lower(), word_vectors)
print(result)

#(0.8259648531138358, 0.7907231007327291, 0.8079598641116739)