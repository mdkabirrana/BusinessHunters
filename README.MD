Requirements:
	- Python 3
	- XGBOOST - (XGBoost)
	- Pandas - (Reading Files)
	- Numpy - (Working with dataset)
	- sklearn.decomposition - (PCA)
	- sklearn.linear_model - (Linear Regression)
	- sklearn.preprocessing - (Polynomial Features)
	- matplotlib.pyplot - (Plotting Graphs)

__________________________________________________________________________________________________________________________________________________________

DATA PRERPROCESSING (2 files):

File 1 (DataPreprocessing_Method1.py):
- Data preprocessing for Methods 1 and 1A.
- Takes 2 files as input:
	- train_v2.csv 
	- test_v2.csv
- Applies Methods 1 and 1A to both files.
- Produces 2 files
	- method1_train_v2.csv
	- method1_test_v2.csv

File 2 (DataPreprocessing_Method2.py):
- Data preprocessing for Methods 2 and 2A.
- Does NOT perform PCA (See File 4)
- Takes 2 files as input:
	- train_v2.csv 
	- test_v2.csv
- Applies Methods 2 and 2A to both files.
- Produces 2 files
	- method2_train_v2.csv
	- method2_test_v2.csv

__________________________________________________________________________________________________________________________________________________________

Modelling (3 files):

File 3 (models_Method1.py):
- Predictive Modelling applied to dataset after preprocessing using Methods 1 and 1A.
- Takes 2 files as input:
	- method1_train_v2.csv
	- method1_test_v2.csv
- Applies 3 models to each method, so each model used twice, once for each type of data; Linear Regression, Polynomial Regression and XGBoost.
- Chooses the model with best result and produces a file (NOT CONSIDERED FOR KAGGLE SUBMISSION):
	- submission.csv

File 4 (models_Method2.py):
- Predictive Modelling applied to dataset after preprocessing using Methods 2 and 2A.
- Performs PCA
- Takes 2 files as input:
	- method2_train_v2.csv
	- method2_test_v2.csv
- Applies 3 models to each method, so each model used twice, once for each type of data; Linear Regression, Polynomial Regression and XGBoost.
- Chooses the model with best result and produces a file (NOT CONSIDERED FOR KAGGLE SUBMISSION):
	- submission.csv

File 5 (best_Model.py):
- Only uses XGBoost for prediction, applied to method 2A
- Takes 2 files as input:
	- method2_train_v2.csv
	- method2_test_v2.csv
- Produces 1 file (CONSIDERED FOR KAGGLE SUBMISSION; PRODUCES BEST ACCURACY):
	- submission.csv

__________________________________________________________________________________________________________________________________________________________

Submission Files (submission.csv)
- Contains 2 columns
	- full Visitor ID
	- Log of transaction revenue


__________________________________________________________________________________________________________________________________________________________


Steps to run correctly:

1. Install all requirements
2. Download datasets (train_v2.csv, test_v2.csv) from the following link: https://www.kaggle.com/c/ga-customer-revenue-prediction/data (REQUIRES KAGGLE ACCOUNT)
3. Execute in any order:
	- DataPreprocessing_Method1.py 
	- DataPreprocessing_Method2.py
4. Execute in any order:
	- models_Method1.py (DEPENDENT ON DataPreprocessing_Method1.py)
	- models_Method2.py (DEPENDENT ON DataPreprocessing_Method2.py)	
	- best_Model.py (DEPENDENT ON DataPreprocessing_Method2.py)

__________________________________________________________________________________________________________________________________________________________

DIVISION OF LABOR:

Everything done together.
	