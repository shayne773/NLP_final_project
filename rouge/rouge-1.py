#overlapping of single words

from collections import Counter
from nltk.tokenize import word_tokenize

#get: lower case generated summary
#ref: lower case reference summary
def rouge_1(ref, get):
    #tokenizing
    ref_tokens = [word for word in word_tokenize(ref)]
    get_tokens = [word for word in word_tokenize(get)]
    
    # counter: each unique words and their counts
    ref_counter = Counter(ref_tokens)
    get_counter = Counter(get_tokens)
    
    # get sum of overlapping words
    overlap = sum((ref_counter & get_counter).values())
    
    #precision: overlap/generated
    #recall: overlap/reference
    precision = overlap / sum(get_counter.values()) if get_counter else 0
    recall = overlap / sum(ref_counter.values()) if ref_counter else 0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, F1

get_summary = "a quick brown fox jumps over the lazy dog"
ref_summary = "the swift fox leaps over a sleeping dog"

result = rouge_1(get_summary.lower(), ref_summary.lower())
print(result)
