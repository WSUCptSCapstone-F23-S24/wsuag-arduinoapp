import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style

# Read data and apply date_time as the index
df = pd.read_csv("./test_results/curve.csv", parse_dates=['date_time'], index_col='date_time')

# Drop rows where 'mean' is NaN and where 'mean' is greater than 1
df = df.dropna(subset=['mean'])  # Remove NaN values
df = df[df['mean'] <= 1]  # Keep rows where 'mean' is less than or equal to 1

# Ensure the DataFrame is sorted by date (if not already)
df.sort_index(inplace=True)

# Set the graph style
style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(14, 5))

# Draw the line with error bars
ax.errorbar(
    x=df.index,
    y=df['mean'],
    yerr=df['std'],  # Add standard deviation as error bar
    fmt='-o',  # Line style with circle markers
    label='Mean NDVI with STD',
    ecolor='red',  # Error bar color
    color='blue',  # Line color
    capsize=3,  # Error bar cap size
)

# Set labels
ax.set_xlabel('Date', size=15)
ax.set_ylabel('95th Percentile of Pixel Value', fontsize=15)

# Set x-tick labels to show actual dates
dates = df.index.strftime('%Y-%m-%d').to_series().unique()
ax.set_xticks(df.index)  # Set x-ticks to match the index
ax.set_xticklabels(labels=dates, rotation=45, ha='right', size=9)

# Optional: Adjust the y-axis scale if needed
# ax.set_ylim(lower_limit, upper_limit)

# Set the legend
ax.legend(ncol=3, frameon=False, loc='lower center')

plt.tight_layout()  # Adjust the layout to make room for the rotated x-tick labels
plt.show()
