import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

# Read data and apply date style as the index
df = pd.read_csv("./test_results/curve.csv", parse_dates=['date'], index_col='date')

# Drop rows where 'mean' is NaN and where 'mean' is greater than 1
df = df.dropna(subset=['mean'])  # Remove NaN values
df = df[df['mean'] <= 1]  # Keep rows where 'mean' is less than or equal to 1

# Ensure the DataFrame is sorted by date (if not already)
df.sort_index(inplace=True)

# Set the graph style
style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(14, 5))

# Draw the line
sns.pointplot(
    x=df.index.dayofyear,
    y=df['mean'],
    errorbar=None,
    scale=0.3)

# Set label
ax.set_xlabel('Date', size=15)
ax.set_ylabel('95th Percentile of Pixel Value', fontsize=15)

# Set x-tick labels with the updated DataFrame to ensure no missing dates and no values over 1
ax.set_xticks(range(len(df.index.dayofyear)))  # Set x-ticks to match the number of points
ax.set_xticklabels(labels=df.index.strftime('%Y-%m-%d').unique(), rotation=45, ha='right', size=9)

# Optional: Adjust the y-axis scale if needed
# ax.set(ylim=(lower_limit, upper_limit))

# Set the legend
legend_handles, _ = ax.get_legend_handles_labels()
ax.legend(legend_handles, ['Your Legend Here'], ncol=3, frameon=False, loc='lower center')

plt.tight_layout()  # Adjust the layout to make room for the rotated x-tick labels
plt.show()
