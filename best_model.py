import pandas as pd
import json
import numpy as np
import math
from math import log
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import matplotlib.pyplot as plt


# Reading processed test and train datasets.
def readDataset(filePath):

    df = pd.read_csv(filePath)

    y = df['transactionRevenue'].values

    importantColumns = ['visitNumber', 'bounces',
                        'hits', 'newVisits', 'pageviews']

    columns = df.columns.values
    for a in columns:
        if df[a].isnull().any().any():
            df.drop([a], axis=1, inplace=True)

    # NEEDED FOR FINAL SUBMISSION
    fullvisitorIDs = df['fullVisitorId'].values

    df.drop(['transactionRevenue', 'fullVisitorId', 'visitId',
             'visitStartTime'], axis=1, inplace=True)

    return df, y, importantColumns, fullvisitorIDs


# METHOD 2A
# Applied Z_NORMALIZATION during preprocessing
# NO PCA
# Returns best prediction
def modelTrainingMethod2A(df_train, df_test, ytrain, ytest, train_importantColumns, test_importantColumns):

    # XGBOOST
    classifier = xgb.XGBClassifier(
        max_depth=3, n_estimators=300, learning_rate=0.05)
    classifier.fit(
        df_train[train_importantColumns].values, ytrain)
    yResult = classifier.predict(
        df_test[test_importantColumns].values)
    return yResult

def writeBestPredictionToFile(yPred, fullvisitorIDs_test):
    
    # The following is the format required by KAGGLE
    df_pred = pd.DataFrame()
    df_pred['fullVisitorId'] = fullvisitorIDs_test
    df_pred['PredictedLogRevenue'] = yPred
    df_pred = df_pred.groupby('fullVisitorId', as_index=False)['PredictedLogRevenue'].sum()
    df_pred['PredictedLogRevenue'] = df_pred['PredictedLogRevenue'].apply(np.log1p)
    df_pred.to_csv('submission.csv', encoding='utf-8', index=False)


def main():
    
    # Train dataset
    filePath = 'method2_train_v2.csv'
    df_train, ytrain, train_importantColumns, fullvisitorIDs_train = readDataset(
        filePath)

    # Test dataset
    filePath = 'method2_test_v2.csv'
    df_test, ytest, test_importantColumns, fullvisitorIDs_test = readDataset(
        filePath)

    # Training WITHOUT PCA
    yPredMethod2A = modelTrainingMethod2A(df_train, df_test, ytrain, ytest,
                                          train_importantColumns, test_importantColumns)

    # writing the submission.csv file
    writeBestPredictionToFile(yPredMethod2A, fullvisitorIDs_test)


if __name__ == "__main__":
    main()
