#ROUGE-1: ~0.4–0.6+ is good.
#ROUGE-2: ~0.2–0.4+ is reasonable.
#ROUGE-L: ~0.2–0.5+ is expected.
#ROUGE-WE-1 : 0.5–0.8+ for good summaries.
from rouge_we import *
from rouge_1 import *
from rouge_2 import *
from rouge_L import *
from gusum import *

import pandas as pd

df = pd.read_csv('data/test.csv')

columns = ['id', 'GUSUM_precision', 'GUSUM_recall', 'GUSUM_F1']

result_1 = pd.DataFrame(columns=columns)
result_2 = pd.DataFrame(columns=columns)
result_L = pd.DataFrame(columns=columns)
result_we = pd.DataFrame(columns=columns)


result_1_file = 'result/result_rouge_1.csv'
result_2_file = 'result/result_rouge_2.csv'
result_L_file = 'result/result_rouge_L.csv'
result_we_file = 'result/result_rouge_we.csv'

# headers
# everytime this file run replace all existing contents
pd.DataFrame(columns=columns).to_csv(result_1_file, index=False)
pd.DataFrame(columns=columns).to_csv(result_2_file, index=False)
pd.DataFrame(columns=columns).to_csv(result_L_file, index=False)
pd.DataFrame(columns=columns).to_csv(result_we_file, index=False)



for i in range(len(df)):
    try:
        g = Graph(document=df['article'][i])

        get_sum = ' '.join([s[0] for s in g.summarize()])

        row_1 = (df['id'][i],) + rouge_1(df['highlights'][i], get_sum)
        row_2 = (df['id'][i],) + rouge_2(df['highlights'][i], get_sum, 2)
        row_L = (df['id'][i],) + rouge_L(df['highlights'][i], get_sum)
        row_we = (df['id'][i],) + rouge_we(df['highlights'][i], get_sum, word_vectors)

        pd.DataFrame([row_1], columns=columns).to_csv(result_1_file, mode='a', index=False, header=False)
        pd.DataFrame([row_2], columns=columns).to_csv(result_2_file, mode='a', index=False, header=False)
        pd.DataFrame([row_L], columns=columns).to_csv(result_L_file, mode='a', index=False, header=False)
        pd.DataFrame([row_we], columns=columns).to_csv(result_we_file, mode='a', index=False, header=False)

    except Exception as e:
        
        print(f"Error processing ID {df['id'][i]} at row {i}: {e}")
        
        continue
