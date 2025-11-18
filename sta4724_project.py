import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
    
def impute_vals(df, fit = None, transform = False):
    missing_loc = df.isna()
   
    # updating missing publisher vals to unknown
    missing = df.loc[missing_loc['publisher'] == True]
    missing.loc[:, 'publisher'] = 'unknown'
    df.update(missing)
    
    missing = df.loc[missing_loc['year'] == True]
    if transform: # set avg release year of each platform for test set
        missing.loc[:,'year'] = missing['platform']
        missing.replace(fit, inplace = True)
        df.update(missing)
    else: # get and set avg release year of each platform for training set
        yr_avgs = df[['year','platform']].groupby('platform').mean()
        np.rint(yr_avgs, out = yr_avgs)
        yr_avgs = yr_avgs.to_dict()
        missing.loc[:,'year'] = missing['platform']
        missing.replace(yr_avgs, inplace = True)
        df.update(missing)

    # set missing years for obs that have year in title
    yr_fmts = r'(19|20)(\d{2})|(?<=\s)([0-2]|[8-9])[0-9](?=\Z|\s)|(?<=2k|2K)(\d+)'
    matches = list(map(lambda x: re.search(yr_fmts, x), missing['name']))
    matches = pd.Series(data = map(lambda x: x.group() if x != None else None,
                                   matches), index = missing.index, name = 'year').dropna()
    matches = matches.astype('int64').sub(1)
    matches.loc[(matches < 100) & (matches > 79)] += 1900
    matches.loc[matches < 79] += 2000
    df.update(matches)
    
    if 2497 in df.index:
        df.at[2497, 'year'] = 2002
    if 12015 in df.index:
        df.at[12015, 'year'] = 2003
    
    if transform:       
        return df
    else:
        return df, yr_avgs
  
def rmv_jp_sales(df):
    titles_loc = df.loc[:, 'name'].str.contains(r'\(jp sales\)', case = False)
    titles_loc = df.loc[titles_loc == True]
    df.drop(index = titles_loc.index, inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    return df
    
def name_as_cat(df):
    names = df['name'].unique()
    mapping = {name: i for i, name in enumerate(names, start = 1)}
    df['name'] = df['name'].map(mapping)
    df['name'] = df['name'].astype('category')
    return df
    
os.chdir('/Users/presleywhitehead/Desktop/sta4724 project')
df = pd.read_csv('./video_games_sales.csv')
df.all()
df.drop(columns = ['other_sales','global_sales', 'rank'], inplace = True)


cats = ['genre','publisher','platform']
for cat in cats:
    print(f'{cat} value counts:\n')
    print(df[cat].value_counts())

X = df.drop(columns = 'na_sales')
y = df.loc[:, 'na_sales']

SEED = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
    random_state = SEED)
X_train, iv_fit = impute_vals(X_train)
X_test = impute_vals(X_test, fit = iv_fit, transform = True)

X_train.drop(columns = 'name', inplace = True)
X_test.drop(columns = 'name', inplace = True)

te = TargetEncoder(target_type='continuous', random_state = SEED)
X_train.loc[:, cats] = te.fit_transform(X_train[cats], y_train)
X_test.loc[:, cats] = te.transform(X_test[cats])

rf = RandomForestRegressor(n_estimators = 100, random_state = SEED)

clf = rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
r_squared = rf.score(X_test, y_test)
mse = MSE(y_test, y_pred)
print(mse)
# actual vs predicted
plt.scatter(y_test, y_pred, marker = '.', c = '#DC8C91')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest Actual vs. Predicted')
plt.show()
plt.clf()

# feature importance
importance = rf.feature_importances_
importance_dct = dict(zip(X_train.columns, importance))
plt.barh(list(importance_dct.keys()), list(importance_dct.values()), color = '#DC8C91')
plt.title('Random Forest Feature Importance')
plt.show()
plt.clf()

# fitted vs residuals
resids = y_test - y_pred
plt.scatter(y_pred, resids, marker = '.', c = '#DC8C91')
plt.xlabel('Fitted')
plt.ylabel('Residuals')
plt.title('Random Forest Fitted vs. Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.show()
plt.clf()



print(list(['2025']*4))

a,b,c = [], [], []
a is b
a.extend([2025]*4)
print(a)
b.extend(list(map(lambda x: x**2, range(5))))
print(b)
x = list(range(2020,2025))
print(x)
