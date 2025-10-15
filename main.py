import os 
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_attri , cat_attri):
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
    return full_pipline

if not os.path.exists(MODEL_FILE):
    housing=pd.read_csv("housing.csv")
    housing["income_cat"]=pd.cut(housing["median_income"] , bins=[0.0 , 1.5 , 3.0 , 4.5 , 6 , np.inf] , labels=[1 ,2 , 3 , 4 , 5])
    split=StratifiedShuffleSplit(n_splits=1 , test_size=0.2 , random_state=42)
    for train_index , test_index in split.split(housing , housing["income_cat"]):
        train_set=housing.loc[train_index].drop("income_cat" , axis=1)
        test_index=housing.loc[test_index].drop("income_cat" , axis=1).to_csv("train_set.csv")

    housing_labels=housing["median_house_value"].copy()
    housing_features=housing.drop("median_house_value" , axis=1)

    num_attri=housing_features.drop("ocean_proximity" , axis=1).columns.to_list()
    cat_attri=["ocean_proximity"]

    pipeline=build_pipeline(num_attri , cat_attri)
    final_housing=pipeline.fit_transform(housing_features)

    model=RandomForestRegressor(random_state=42)
    model.fit(final_housing , housing_labels)
    
    joblib.dump(model , MODEL_FILE)
    joblib.dump(pipeline , PIPELINE_FILE)

    print("congrats the model is trained")

else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    input_data=pd.read_csv("train_set.csv")
    transformed_input= pipeline.transform(input_data)
    predictions=model.predict(transformed_input)
    input_data["median_house_value"]=predictions

    input_data.to_csv("output.csv" , index=False)


    else:
    model=MODEL_FILE
    pipeline=Pipeline_FILE
    input_data=pd.read_csv("testset.csv")
    transformes_input=pipeline.transform(input_data)
    predictions=model.predict(transformes_input)
    input_data["charges"]=predictions

    input_data.to_csv("output.csv")



