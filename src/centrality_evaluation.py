'''
this file is for evaluating the different weights of forward edges and backward edges
weights range from 0 to 1
the relationship between forward and backward weights in the evaluation is:
backward_weight = 1-forward_weight
'''
from rouge_1 import *
from rouge_2 import *
from rouge_L import *
from gusum import *

import pandas as pd

# Define weights
forward_weights = [i * 0.1 for i in range(11)]
backward_weights = [1 - weight for weight in forward_weights]

df = pd.read_csv('data/test.csv')

columns = ['forward_weight', 'backward_weight', 'rouge1_f1_mean', 'rouge2_f1_mean', 'rougeL_f1_mean']

result_file = './results/diversified_centrality_results.csv'

# Ensure the result directory exists
import os
if not os.path.exists('./results'):
    os.makedirs('./results')

# Create or overwrite the result file with headers
pd.DataFrame(columns=columns).to_csv(result_file, index=False)

# Main execution
for fw, bw in zip(forward_weights, backward_weights):
    print(f"Calculating results for forward weight: {fw}, backward weight: {bw}")

    # Temporary DataFrame to store results for this weight configuration
    results_df = pd.DataFrame(columns=['rouge1_f1', 'rouge2_f1', 'rougeL_f1'])

    for i in range(len(df)):
        try:
            print(f'Processing ID: {df["id"][i]}')
            g = Graph(document=df['article'][i], forward_weight=fw, backward_weight=bw)

            get_sum = ' '.join([s[0] for s in g.summarize_with_diversity()])

            row_1 = rouge_1(df['highlights'][i], get_sum)
            row_2 = rouge_2(df['highlights'][i], get_sum, 2)
            row_L = rouge_L(df['highlights'][i], get_sum)

            # Store results for this row
            results_df.loc[len(results_df)] = [row_1[2], row_2[2], row_L[2]]

        except Exception as e:
            print(f"Error processing ID {df['id'][i]} at row {i}: {e}")
            continue

    # Calculate means for this weight configuration
    mean_rouge1_f1 = results_df['rouge1_f1'].mean()
    mean_rouge2_f1 = results_df['rouge2_f1'].mean()
    mean_rougeL_f1 = results_df['rougeL_f1'].mean()

    # Append results for this weight configuration to the final CSV file
    pd.DataFrame([[fw, bw, mean_rouge1_f1, mean_rouge2_f1, mean_rougeL_f1]], columns=columns).to_csv(result_file, mode='a', index=False, header=False)

    print(f"Appended results for forward_weight={fw}, backward_weight={bw}")

print("Results saved to ./results/centrality_results.csv")