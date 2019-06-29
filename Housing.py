#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from six.moves import urllib

import pandas as pd

HOUSING_PATH = "datasets/housing"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[2]:


housing = load_housing_data()
housing.head()


# In[3]:


housing.info()


# In[4]:


housing.describe()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(10,7))
plt.show()


# In[6]:


import numpy as np

def split_train_set(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_set(housing, 0.2)
print("Train: ", len(train_set), "+ Test: ", len(test_set))


# In[7]:


import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# In[8]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[9]:


housing.hist(column='median_income', bins=10)
plt.show()


# In[10]:


housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing.hist('income_cat', bins=10)
plt.show()

housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
housing.hist('income_cat')
plt.show()


# In[11]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]
    
for set in (strat_train_set, strat_test_set):
    set.drop(columns='income_cat', inplace=True)


# In[12]:


housing = strat_train_set.copy()
housing.describe()


# In[13]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[14]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2,
             s=housing["population"]/100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)


# In[15]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[19]:


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[20]:


attributes = ["median_house_value", "households", "total_bedrooms", "population"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[21]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# In[27]:


housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[44]:


housing = strat_train_set.drop(columns="median_house_value")
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop(columns="ocean_proximity")
# imputer.fit(housing_num)
# print(imputer.statistics_)
# X = imputer.transform(housing_num)
X = imputer.fit_transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# housing_tr.describe()


# In[77]:


# housing["ocean_proximity"].value_counts()

# from sklearn.preprocessing import LabelEncoder

# encoder = LabelEncoder()
# housing_cat_encoded = encoder.fit_transform(housing["ocean_proximity"])

# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder()
# # print(housing["ocean_proximity"].to_numpy().reshape(-1,1).shape)
# housing_cat_1hot = encoder.fit_transform(housing["ocean_proximity"].to_numpy().reshape(-1,1))
# print(type(housing_ocean_cat))

from sklearn.preprocessing import LabelBinarizer

# encoder = LabelBinarizer()
encoder = LabelBinarizer(sparse_output=True)
# housing_cat_1hot = encoder.fit_transform(housing["ocean_proximity"])
print(type(housing_cat_1hot))


# In[93]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_extra_attribs = pd.DataFrame(housing_extra_attribs)

print(type(housing.columns))


# In[ ]:





# In[ ]:




