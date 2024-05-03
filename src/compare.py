import pandas as pd


df1 = pd.read_csv("path1")
df2 = pd.read_csv("path2")
df1.set_index(['date', 'time', 'sprectrum', 'rep_pic'])
df2.set_index(['date', 'time', 'sprectrum', 'rep_pic'])
print(df1)

df_res = df1["mean"].sub(df2["mean"], fill_value=-1111111111111111111)


#pd.merge(df1, df2,  how='left', left_on=['date', 'time', 'sprectrum', 'rep_pic'], right_on = ['date', 'time', 'sprectrum', 'rep_pic'])

#date	time	sprectrum	rep_pic



# df_res[df_res.groupby(level=0).transform('nunique').gt(1)]

# df_res['mean'] = df_res.groupby(['date', 'time', 'sprectrum', 'rep_pic'])['mean'].diff()

df_res.to_csv('./out.csv',index=True)
