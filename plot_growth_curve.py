#original code by  Worasit Sangjan

import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
 
 

 
# Read data and apply date style as the index
df = pd.read_csv("C:\\Users\\code8\\Downloads\\New folder (6)\\curve.csv",
                 parse_dates=['date'], index_col='date')
df.info()
 
# Select the targeted parameters
var = 5
vi = 'ndvi'
y = 'mean'
df1 = df
 
# Set the graph style
style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(14, 5))
 
# Draw the line
sns.pointplot(
    x=df1.index.dayofyear,
    y=df1[y],
    errorbar=None,
    scale=0.3)
 
# Set label
ax.set_xlabel('Date', size=15)
ax.set_ylabel('95th Percentile of Pixel Value  ', fontsize=15)
# Set x-tick label
ax.set_xticklabels(labels=df.index.strftime('%Y-%m-%d').sort_values().unique(),
                   rotation=45,
                   ha='right',
                   size=9)
# Set the scale of y-axis
# ax.set(ylim=(1, 3))
# Set the legend
legend_handles, _= ax.get_legend_handles_labels()
ax.legend(legend_handles, ['Blue', 'Green', 'Red'], ncol=3, frameon=False, loc='lower center',)
plt.show()