import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

housing=pd.read_csv("housing.csv")

housing["income_cat"]=pd.cut(housing["median_income"],
                             bins=[0.0 , 1.5 , 3 , 4.5 , 6 , np.inf,],
                            labels=[1 ,2 ,3 ,4 ,5])

split=StratifiedShuffleSplit(n_splits=1 , test_size=0.2 , random_state=42)
for train_index , test_index in split.split(housing ,housing["income_cat"]):
    train_set=housing.loc[train_index].drop("income_cat" , axis=1)
    test_set=housing.loc[test_index].drop("income_cat" , axis=1)

housing=train_set.copy()

housing_labels=housing["median_house_value"]
housing=housing.drop("median_house_value" , axis=1)

num_attri=housing.drop("ocean_proximity" , axis=1).columns.tolist()
cat_attri=["ocean_proximity"]

num_pipelines=Pipeline([
    ("imputer" , SimpleImputer(strategy="median")),
    ("scaler" , StandardScaler()),
])

cat_pipeline=Pipeline([
    ("onehot" , OneHotEncoder(handle_unknown="ignore"))
])

full_pipline=ColumnTransformer([
    ("num" , num_pipelines , num_attri),
    ("cat" , cat_pipeline , cat_attri),
])

final_housing=full_pipline.fit_transform(housing)

print(final_housing.shape)

lin_reg=LinearRegression()
lin_reg.fit(final_housing , housing_labels)
lin_predicts=lin_reg.predict(final_housing)
lin_rmse=-cross_val_score(lin_reg , final_housing , housing_labels , scoring="neg_root_mean_squared_error" , cv=10)
print(pd.Series(lin_rmse).describe())

dec_tree_reg=DecisionTreeRegressor(random_state=42)
dec_tree_reg.fit(final_housing , housing_labels)
dec_tree_prd=dec_tree_reg.predict(final_housing)
dec_tree_rmse=-cross_val_score(dec_tree_reg , final_housing , housing_labels , scoring="neg_root_mean_squared_error" , cv=10) 
print(pd.Series(dec_tree_rmse).describe())

ran_forest_reg=RandomForestRegressor(random_state=42)
ran_forest_reg.fit(final_housing , housing_labels)
ran_forest_pred=ran_forest_reg.predict(final_housing)
ran_forest_rmse=cross_val_score(ran_forest_reg, final_housing , housing_labels , scoring="neg_root_mean_squared_error" , cv=10)
print(pd.Series(ran_forest_rmse).describe())