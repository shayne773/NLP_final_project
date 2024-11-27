#Longest common subsequence

from nltk import word_tokenize


#find lcs for two texts
def longest_common_subsequence(text_a, text_b):

    m, n = len(text_a), len(text_b)
    temp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for a in range(1, m + 1):
        for b in range(1, n + 1):
            if text_a[a - 1] == text_b[b - 1]:
                temp[a][b] = temp[a - 1][b - 1] + 1
            else:
                temp[a][b] = max(temp[a - 1][b], temp[a][b - 1])

    return temp[a][b]


#get: lower case generated summary
#ref: lower case reference summary
def rouge_L(ref, get):
 
    LCS_length = longest_common_subsequence(ref,get)

    
    #precision: longest common s length/len(get)
    #recall: longest common s length/len(ref)

    precision = LCS_length / len(get) if len(get) != 0 else 0
    recall = LCS_length / len(ref) if len(ref) != 0 else 0
    F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, F1

get_summary = "The cat sat on the mat."
ref_summary = "The cat lay on the mat."

result = rouge_L(get_summary.lower(), ref_summary.lower())
print(result)

