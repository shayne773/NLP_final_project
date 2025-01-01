import matplotlib.pyplot as plt
import pandas as pd

# Original results (no diversity)
data_no_div = {
    "forward_weight": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "rouge1_f1_mean": [
        0.264439775908654,
        0.2668282502983513,
        0.269922883258818,
        0.2830094765000819,
        0.31595676662446404,
        0.34636621433787873,
        0.36364656663694345,
        0.37087876475099907,
        0.374651509027112,
        0.37711383300656587,
        0.3788633974332613,
    ],
}

# Results with diversity
data_div = {
    "forward_weight": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "rouge1_f1_mean": [
        0.26615848422649907,
        0.268436268829434,
        0.2718717451000043,
        0.2853490647680157,
        0.31802370391808904,
        0.34954519332290424,
        0.3668612611731285,
        0.3746107763030641,
        0.37838313232131565,
        0.380475430086784,
        0.3819877170059368,
    ],
}

# Convert to DataFrames
df_no_div = pd.DataFrame(data_no_div)
df_div = pd.DataFrame(data_div)

# Plot both lines on the same figure
plt.figure(figsize=(10, 6))
plt.plot(df_no_div["forward_weight"], df_no_div["rouge1_f1_mean"], marker="o", label="No Diversity", color='orange')
plt.plot(df_div["forward_weight"], df_div["rouge1_f1_mean"], marker="o", label="With Diversity", color='blue')

# Add title and labels
plt.title("ROUGE-1 F1 Score vs Backward Weight")
plt.xlabel("Backward Weight")
plt.ylabel("ROUGE-1 F1 Mean")
plt.grid(True)

# Add legend to distinguish the two lines
plt.legend()

# Show the plot
plt.show()