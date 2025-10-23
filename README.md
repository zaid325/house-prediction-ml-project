This project predicts house prices based on various features such as location, area, number of rooms, and other parameters.
It demonstrates a complete end-to-end Machine Learning workflow — from data preprocessing to model building, pipeline creation, and evaluation.


Features
Data Cleaning & Preprocessing — handled missing values, outliers, and categorical encoding

Feature Engineering — selected and transformed relevant features for model input

 ML Pipeline Creation — automated preprocessing and model training steps using scikit-learn pipelines

 Model Building — trained and evaluated a Random Forest Regressor for robust price predictions

 Model Evaluation — assessed performance using metrics such as R² Score and Mean Absolute Error

 Model Saving — final model saved as a .pkl file for deployment or inference


 Tech Stack
Python 3.x

NumPy, Pandas — for data cleaning and analysis

Matplotlib, Seaborn — for visualization

Scikit-learn — for pipeline creation and Random Forest model

Joblib / Pickle — for model serialization

📦 house-value-prediction
 ┣ 📂 data
 ┃ ┗─ house_data.csv
 ┣ 📂 notebooks
 ┃ ┗─ EDA_and_Model.ipynb
 ┣ 📂 src
 ┃ ┣─ data_preprocessing.py
 ┃ ┣─ model_training.py
 ┃ ┗─ pipeline.py
 ┣ 📜 model.pkl  (ignored in Git)
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┗ 📜 app.py  (optional — for deployment)
