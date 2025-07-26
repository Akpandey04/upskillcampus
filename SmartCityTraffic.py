#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import holidays
import lightgbm as lgb
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


train = pd.read_csv('train_aWnotuB.csv')
test = pd.read_csv('test_BdBKkAj.csv')


# In[9]:


train.head()


# In[10]:


train.info()


# In[11]:


train.describe()


# In[ ]:


sns.set_style("whitegrid")
plt.figure(figsize=(15, 7))
sns.lineplot(data=train[train['Junction']==1], x='DateTime', y='Vehicles')
plt.title('Traffic Volume Over Time (Junction 1)')
plt.show()


# In[202]:


def create_time_features(df):
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['hour'] = df['DateTime'].dt.hour
    df['day'] = df['DateTime'].dt.day
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    df['dayofyear'] = df['DateTime'].dt.dayofyear
    df['weekofyear'] = df['DateTime'].dt.isocalendar().week.astype(int)
    india_holidays = holidays.country_holidays('IN', years=[2015, 2016, 2017])
    df['is_holiday'] = df['DateTime'].dt.date.apply(lambda x: x in india_holidays).astype(int)
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x in [5, 6] else 0)

    return df


# In[203]:


train = create_time_features(train)
test = create_time_features(test)


# In[204]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='Junction', y='Vehicles', data=train, palette='magma')
plt.title('Traffic Volume Distribution by Junction')
plt.show()


# In[205]:


plt.figure(figsize=(10, 6))
sns.barplot(x='is_holiday', y='Vehicles', data=train, palette='coolwarm')
plt.title('Average Traffic on Holidays vs. Regular Days')
plt.xticks([0, 1], ['Regular Day', 'Holiday'])
plt.show()


# In[206]:


train = pd.get_dummies(train, columns=['Junction'])
test = pd.get_dummies(test, columns=['Junction'], prefix='Junction')


# In[207]:


train = train.sort_values('DateTime')
split_date = '2017-04-01'
val_df = train[train['DateTime'] >= split_date]
train_df = train[train['DateTime'] < split_date]


# In[208]:


features = [col for col in train.columns if col not in ['ID', 'DateTime', 'Vehicles']]
target = 'Vehicles'


# In[209]:


X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]


# In[210]:


lgbm = lgb.LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)
lgbm_preds = lgbm.predict(X_val)


# In[211]:


rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_val)


# In[212]:


print("LightGBM R2 Score:", r2_score(y_val, lgbm_preds))
print("LightGBM RMSE:", np.sqrt(mean_squared_error(y_val, lgbm_preds)))
print("-" * 20)
print("Random Forest R2 Score:", r2_score(y_val, rf_preds))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_val, rf_preds)))
print("-" * 20)


# In[213]:


final_model = rf


# In[214]:


final_model.fit(train[features], train[target])


# In[215]:


joblib.dump(final_model, 'traffic_model_rf.pkl')


# In[216]:


test_preds = final_model.predict(test[features])
submission = pd.DataFrame({'ID': test['ID'], 'Vehicles': test_preds})
submission['Vehicles'] = submission['Vehicles'].apply(lambda x: int(max(0, x))) # Make sure vehicles are positive integers
submission.to_csv('submission.csv', index=False)

