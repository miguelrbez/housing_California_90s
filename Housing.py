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
# housing = 
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


# In[16]:


from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[17]:


attributes = ["median_house_value", "households", "total_bedrooms", "population"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[18]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# In[19]:


housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[20]:


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


# In[21]:


# housing["ocean_proximity"].value_counts()

# from sklearn.preprocessing import LabelEncoder

# encoder = LabelEncoder()
# housing_cat_encoded = encoder.fit_transform(housing["ocean_proximity"])

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
# print(housing["ocean_proximity"].to_numpy().reshape(-1,1).shape)
housing_cat_1hot = encoder.fit_transform(housing["ocean_proximity"].to_numpy().reshape(-1,1))
print(type(housing_cat_1hot))
print(encoder.categories_)
# dfgdh = DataFrameSelector(["ocean_proximity"]).fit_transform(housing)
# print(np.unique(dfgdh))


# from sklearn.preprocessing import LabelBinarizer

# encoder = LabelBinarizer()
# housing_cat_1hot = encoder.fit_transform(housing["ocean_proximity"])
# print(housing_cat_1hot)

# housing_cat_1hot = pd.get_dummies(housing, columns=['ocean_proximity'])
# housing_cat_1hot.head()


# In[22]:


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
housing_extra_attribs.head()
# print(type(housing.columns))


# In[23]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
# selector = DataFrameSelector(["ocean_proximity"])
# housing_cat_select = selector.fit_transform(housing)
# print(housing_cat_select[:5])

# encoder = LabelBinarizer()
# housing_cat_1hot = encoder.fit_transform(housing_cat_select)
# print(housing_cat_1hot[:5])


# In[24]:


from sklearn.preprocessing import OneHotEncoder

class CatOneHotEncoder (BaseEstimator, TransformerMixin):
#     from sklearn.preprocessing import OneHotEncoder
    def __init__(self, sparse = False):
        self.sparse = sparse
        encoder = OneHotEncoder(sparse=self.sparse)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        encoder = OneHotEncoder(sparse=self.sparse)
        return encoder.fit_transform(X.reshape(-1,1))
    def get_categories(self, X):
        return list(np.unique(X))
#     def categories_(self, X):
# #         encoder = OneHotEncoder(sparse=self.sparse)
# #         encoder.fit_transform(X.reshape(-1,1))
#         return ["encoder.categories_"]

encoder = CatOneHotEncoder()
encoder.fit(DataFrameSelector(['ocean_proximity']).fit_transform(housing))
# print(encoder.fit_transform(DataFrameSelector(['ocean_proximity']).fit_transform(housing)))
print(encoder.get_categories(housing['ocean_proximity']))
# selector = DataFrameSelector(["ocean_proximity"])
# housing_cat_select = selector.fit_transform(housing)
# print(housing_cat_select[:5])

# encoder = CatEncoder()
# housing_cat_1hot = encoder.fit_transform(housing_cat_select)
# print(type(housing_cat_1hot))


# In[84]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(list(housing_num))),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

# housing_prepared = num_pipeline.fit_transform(housing)
# print(housing_prepared[:5])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('1hot_encoder', CatOneHotEncoder())
])


# housing_prepared = cat_pipeline.fit_transform(housing)
# print(housing_prepared[:5])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

# pipeline = Pipeline([
#     ('selector', DataFrameSelector(list(housing_cat_1hot))),
#     ('imputer', SimpleImputer(strategy="median")),
#     ('attribs_adder', CombinedAttributesAdder()),
#     ('std_scaler', StandardScaler()),
# ])

housing_prepared = full_pipeline.fit_transform(housing)

# print(type(housing_prepared))
# print(housing_prepared[:5])
pd.DataFrame(housing_prepared).head()


# In[27]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# some_data = housing.iloc[:10]
# some_labels = housing_labels[:10]
# some_data_prepared = full_pipeline.fit_transform(some_data)

# print(pd.DataFrame(some_data_prepared[:5]))
# print("Predictions:\"", lin_reg.predict(some_data_prepared))
# print(some_data_prepared.shape)
# print(housing_prepared.shape)

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


# In[28]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
print(tree_rmse)


# In[29]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
# print(rmse_scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
display_scores(tree_rmse_scores)


# In[30]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[31]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# In[33]:


print(np.sqrt(mean_squared_error(forest_reg.predict(housing_prepared), housing_labels)))


# In[41]:


from sklearn.model_selection import GridSearchCV

# param_grid = [
#     {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]},
#     {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2, 4, 6]}
# ]

param_grid = [
    {'n_estimators':[30, 100], 'max_features':[6, 8, 10]}
]

forest_reg = RandomForestRegressor()
     
grid_search = GridSearchCV(forest_reg, param_grid,
                           cv=5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)


# In[34]:


# For quick use when restarting the kernel

from sklearn.model_selection import GridSearchCV

# param_grid = [
#     {'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]},
#     {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2, 4, 6]}
# ]

param_grid = [
    {'n_estimators':[100], 'max_features':[6]}
]

forest_reg = RandomForestRegressor()
     
grid_search = GridSearchCV(forest_reg, param_grid,
                           cv=5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)


# In[35]:


print(np.sqrt(-grid_search.best_score_))
print(grid_search.best_params_)


# In[44]:


cvres = grid_search.cv_results_

for mean_score, params in sorted(list(zip(cvres['mean_test_score'], cvres['params'])), reverse=True):
    print(np.sqrt(-mean_score), params)


# In[36]:


feature_importances = grid_search.best_estimator_.feature_importances_
# print(feature_importances)

extra_attribs = ["rooms_per_household", "pop_per_household", "bedrooms_per_room"]
cat_one_hot_attribs = encoder.get_categories(housing['ocean_proximity'])

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

attribs_importance_list = sorted(list(zip(feature_importances, attributes)), reverse=True)
for element in attribs_importance_list:
    print(element)
    
_, attribs_importance_order = zip(*attribs_importance_list)
attribs_importance_order = list(attribs_importance_order)
print(attribs_importance_order)


# In[37]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop(columns="median_house_value")
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(final_rmse)


# In[73]:


from sklearn.svm import SVR

SupportVectorMachine = SVR()

svr_param_grid = [
    {'kernel':["linear"], 'C':[0.3, 1, 3]},
    {'kernel':["rbf"], 'C':[1, 3], 'gamma':[0.03, 0.1]}
]

svr_grid_search = GridSearchCV(SupportVectorMachine, svr_param_grid,
                               cv=5, scoring="neg_mean_squared_error")

svr_grid_search.fit(housing_prepared, housing_labels)


# In[84]:


svr_cv_results = svr_grid_search.cv_results_
# for sorted(score, param in svr_cv_results['mean_test_score'], svr_cv_results['params'], reverse=True):
#     print(score, param)
for element in sorted(list(zip(np.sqrt(-svr_cv_results['mean_test_score']), svr_cv_results['params']))):
    print(element)
# print(svr_cv_results['params'])


# In[85]:


svr_param_grid = [
    {'kernel':["linear"], 'C':[3, 10, 30, 100]},
    {'kernel':["rbf"], 'C':[10, 3], 'gamma':[3, 1]}
]

svr_grid_search = GridSearchCV(SupportVectorMachine, svr_param_grid,
                               cv=5, scoring="neg_mean_squared_error")

svr_grid_search.fit(housing_prepared, housing_labels)


# In[87]:


svr_cv_results = svr_grid_search.cv_results_
# for sorted(score, param in svr_cv_results['mean_test_score'], svr_cv_results['params'], reverse=True):
#     print(score, param)
for element in sorted(list(zip(np.sqrt(-svr_cv_results['mean_test_score']), svr_cv_results['params']))):
    print(element)
# print(svr_cv_results['params'])


# In[38]:


housing_reduced = pd.DataFrame(housing_prepared)
housing_reduced.columns = attributes
housing_reduced = housing_reduced[attribs_importance_order[:8]]
housing_reduced.head()

# housing_prepared_small = housing_prepared[attribs_importance_order[:8]]
# attribs_importance_order[:8]
# pd.DataFrame(housing_prepared).head()

# print(type(housing_prepared))


# In[39]:


class ReduceFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, attribs, attribs_order, num=8):
        self.attribs = attribs
        self.attribs_order = attribs_order
        self.num = num
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None)      :
        X_dataframe = pd.DataFrame(X)
        X_dataframe.columns = self.attribs
        X_reduced = X_dataframe[self.attribs_order[:(self.num)]]
        return X_reduced.values
    
reducer = ReduceFeatures(attributes, attribs_importance_order)
housing_reduced = reducer.fit_transform(housing_prepared)
print(housing_reduced[:5])

# print(housing_reduced.head())
# housing_reduced = housing_reduced[attribs_importance_order[:8]]
# print(housing_reduced[:5])


# In[40]:


reducer_pipeline = Pipeline([
    ('full_pipeline', full_pipeline),
    ('reducer', ReduceFeatures(attributes, attribs_importance_order))
])

housing_reduced = reducer_pipeline.fit_transform(housing)
print(housing_reduced[:5])


# In[70]:


# param_grid = [
#     {'n_estimators':[30, 100]}
# ]

param_grid = [
    {'n_estimators':[30, 100]}
]
     
grid_search_red = GridSearchCV(forest_reg, param_grid,
                           cv=5, scoring='neg_mean_squared_error')

grid_search_red.fit(housing_reduced, housing_labels)

cvres_red = grid_search_red.cv_results_

for mean_score, params in sorted(list(zip(cvres_red['mean_test_score'], cvres_red['params'])), reverse=True):
    print(np.sqrt(-mean_score), params)


# In[87]:


final_model_red = grid_search_red.best_estimator_

X_test = strat_test_set.drop(columns="median_house_value")
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = reducer_pipeline.transform(X_test)

final_predictions = final_model_red.predict(X_test_prepared)

final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(final_rmse)


# In[88]:


final_model_red = RandomForestRegressor(n_estimators=100)
final_model_red.fit(reducer_pipeline.fit_transform(housing), housing_labels)

X_test = strat_test_set.drop(columns="median_house_value")
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = reducer_pipeline.transform(X_test)

final_predictions = final_model_red.predict(X_test_prepared)

final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print(final_rmse)


# In[ ]:





# In[41]:


list(attributes)


# In[95]:


class HouseValueQuirksElimination(BaseEstimator, TransformerMixin):
    def __init__(self, prices):
        self.prices = prices.to_numpy()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for i in [500000, 450000, 350000]:
            if i == 500000:
                ix, = np.where(self.prices==i)
                print(len(ix))
            else:
                ix_i, = np.where(self.prices==i)
                print(len(ix_i))
                ix = np.append(ix, ix_i)
        return np.delete(X, ix, 0)

quirks_remover = HouseValueQuirksElimination(housing_labels)
# housing_quirks = quirks_remover.fit_transform(housing)


# In[117]:


some_data = DataFrameSelector(list(housing)).fit_transform(housing)
selector = housing_labels.to_numpy()
# print(type(selector))
# print(some_data[:5])
# i, = np.where(selector==500000)
# print(len(i))
# print(type(i))
# print(i.shape)

# ix = np.empty(1)
# print(type(ix))
# print(ix.shape)
# print(ix)

# del ix
for i in [500001, 500000, 450000, 350000]:
    if i == 500000:
        ix, = np.where(selector==i)
        print(len(ix))
    else:
        ix_i, = np.where(selector==i)
        print(len(ix_i))
        ix = np.append(ix, ix_i)

print(len(ix))

# if ix exist:
#     print("Is")
# else:
#     print("Not")
         

some_data_prepared = np.delete(some_data, ix, 0)
print(some_data.shape)
print(some_data_prepared.shape)

# plot()


# In[131]:


print(strat_train_set['median_house_value'].value_counts().head(10))
quirks = [500001, 450000, 350000]
strat_train_set_quirks = strat_train_set[~strat_train_set['median_house_value'].isin(quirks)]
strat_test_set_quirks = strat_test_set[~strat_test_set['median_house_value'].isin(quirks)]
# strat_train_set_quirks.describe()
# strat_test_set_quirks.describe()

# print(~strat_train_set['median_house_value'].isin([500000, 450000, 350000]))

# print(type(strat_train_set.index[strat_train_set['median_house_value'==500000]]))
# df.index[df['BoolCol'] == True].tolist()

# idx = strat_train_set[1].index
# print(idx)

# print(strat_test_set[strat_test_set['median_house_value'].isin([500000])])

# strat_train_set.head()


# In[136]:


housing_quirks = strat_train_set_quirks.drop(columns='median_house_value')
housing_labels_quirks = strat_train_set_quirks['median_house_value'].copy()

housing_quirks_prepared = full_pipeline.fit_transform(housing_quirks)

forest_reg_quirks = RandomForestRegressor()

param_grid = [
    {'n_estimators':[100], 'max_features':[6, 8, 10]}
]

grid_search_quirks = GridSearchCV(forest_reg_quirks, param_grid,
                           cv=5, scoring='neg_mean_squared_error')
grid_search_quirks.fit(housing_quirks_prepared, housing_labels_quirks)

cvres_quirks = grid_search_quirks.cv_results_

for mean_score, params in sorted(list(zip(cvres_quirks['mean_test_score'], cvres_quirks['params'])), reverse=True):
    print(np.sqrt(-mean_score), params)


# In[153]:


final_model_quirks = grid_search_quirks.best_estimator_

X_test = strat_test_set_quirks.drop(columns='median_house_value')
y_test = strat_test_set_quirks['median_house_value']

X_test_prepared = full_pipeline.transform(X_test)
predictions_quirks = final_model_quirks.predict(X_test_prepared)

rmse = np.sqrt(mean_squared_error(y_test, predictions_quirks))
print("rmse = ", rmse)


# In[ ]:




