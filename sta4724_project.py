import pandas as pd
import numpy as np
import os, re
    
def impute_vals(df):
    missing_loc = df.isna()
    for col in df.columns:
        missing = df.loc[missing_loc[col] == True]
        if missing.empty:
            continue
        
        if col == 'year':
            yr_avgs = df[['year','platform']].groupby('platform').mean()
            np.rint(yr_avgs, out = yr_avgs)
            yr_avgs = yr_avgs.to_dict()
            missing.loc[:,'year'] = missing['platform']
            missing.replace(yr_avgs, inplace = True)
            df.update(missing)
 
            yr_fmts = r'(19|20)(\d{2})|(?<=\s)([0-2]|[8-9])[0-9](?=\Z|\s)|(?<=2k|2K)(\d+)'
            matches = list(map(lambda x: re.search(yr_fmts, x), missing['name']))
            matches = pd.Series(data = map(lambda x: x.group() if x != None else None,
                                           matches), index = missing.index, name = 'year').dropna()
            matches = matches.astype('int64').sub(1)
            matches.loc[(matches < 100) & (matches > 79)] += 1900
            matches.loc[matches < 79] += 2000
            df.update(matches)
            
            df.at[2497, 'year'] = 2002
            df.at[12015, 'year'] = 2003
        
        if col == 'publisher':
            missing.loc[:, col] = 'unknown'
            df.update(missing)
            
    return df
            
os.chdir('/Users/presleywhitehead/Desktop/sta4724 project')
df = pd.read_csv('./video_games_sales.csv')

df.drop(columns = ['eu_sales','jp_sales', 'other_sales','global_sales'], inplace = True)
df = impute_vals(df)
df.all()
