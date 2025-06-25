import pandas as pd
import matplotlib.pyplot as plt

df_in_sample = pd.read_csv("data/in_sample_results.csv")
df_out_sample = pd.read_csv("data/out_of_sample_results.csv")

# plot all columns in df
def plot_df(df, title):
    for col in df.columns:
        plt.plot(df[col], label=col)
    plt.xlabel("Time Step")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.show()

plot_df(df_in_sample, title="In-Sample Results of Battery Arbitrage Strategies")
plot_df(df_out_sample, title="Out-of-Sample Results of Battery Arbitrage Strategies")