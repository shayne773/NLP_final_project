#Overlap of consecutive word pairs

from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter


#get: lower case generated summary
#ref: lower case reference summary
#n: n for n-grams

def rouge_2(get, ref, n):
    #tokenizing
    ref_tokens = [word for word in word_tokenize(ref)]
    get_tokens = [word for word in word_tokenize(get)]

    #get n-grams
    get_ngram = Counter(list(ngrams(get_tokens, n)))
    ref_ngram = Counter(list(ngrams(ref_tokens, n)))

    
    # counter: each unique words and their counts
    ref_counter = Counter(ref_ngram)
    get_counter = Counter(get_ngram)

    
    # get sum of overlapping words
    overlap = sum((ref_counter & get_counter).values())
    
    #precision: overlap/generated
    #recall: overlap/reference
    precision = overlap / sum(get_counter.values()) if get_counter else 0
    recall = overlap / sum(ref_counter.values()) if ref_counter else 0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, F1

get_summary = "The cat sat on the mat."
ref_summary = "The cat lay on the mat."

#get_summary = "a quick brown fox jumps over the lazy dog"
#ref_summary = "the swift fox leaps over a sleeping dog"

result = rouge_2(get_summary.lower(), ref_summary.lower(),2)
print(result)


'''
text = 'a quick brown fox jumps over the lazy dog'
tokens = word_tokenize(text)
bigrams = list(ngrams(tokens, 2))
print(bigrams)
'''
