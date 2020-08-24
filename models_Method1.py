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

    y = [math.log(a + 1, 10) for a in df['transactionRevenue'].values]

    importantColumns = ['visitNumber', 'bounces',
                        'hits', 'newVisits', 'pageviews']

    # Columns to be used for plotting
    xTestVisitNumber = df[importantColumns].values[:, 0]
    xTestBounces = df[importantColumns].values[:, 1]
    xTestHits = df[importantColumns].values[:, 2]
    xTestNewVisits = df[importantColumns].values[:, 3]
    xTestPageViews = df[importantColumns].values[:, 4]

    plotColumnsList = [xTestVisitNumber, xTestBounces,
                       xTestHits, xTestNewVisits, xTestPageViews]

    columns = df.columns.values
    for a in columns:
        if df[a].isnull().any().any():
            df.drop([a], axis=1, inplace=True)

    # NEEDED FOR FINAL SUBMISSION
    fullvisitorIDs = df['fullVisitorId'].values

    df.drop(['transactionRevenue', 'fullVisitorId', 'visitId',
             'visitStartTime'], axis=1, inplace=True)

    return df, y, plotColumnsList, fullvisitorIDs


def plotGraph(x, y, yPred, ylabel, title, modelName):
    plt.scatter(x, y,  color='black')
    plt.plot(np.unique(x), np.poly1d(np.polyfit(
        x, yPred, 1))(np.unique(x)))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend([modelName, 'Original Data'], loc='upper left')
    plt.axis([np.amin(x),
              np.amax(x), np.amin(y), np.amax(y)])

    plt.show()
    return


# METHOD 1
# specific columns
def modelTrainingMethod1(df_train, df_test, ytrain, ytest, train_plotColumnsList, test_plotColumnsList):

    # LINEAR REGRESSSION
    result = LinearRegression().fit(df_train.values, ytrain)
    yResult = result.predict(df_test)
    res = np.linalg.norm(yResult - ytest) / \
        math.sqrt(len(df_test.values))
    print("Validation score of Linear Regression: ", res)


# POLYNOMIAL REGRESSSION
    degree = 4
    poly = PolynomialFeatures(degree=degree)
    poly_x = poly.fit_transform(df_train)
    regressor = LinearRegression()
    regressor.fit(poly_x, ytrain)
    yResult2 = regressor.predict(poly.fit_transform(df_test))
    res2 = np.linalg.norm(yResult2 - ytest) / \
        math.sqrt(len(df_test.values))

    print("Validation score of Polynomial Regression: ", res2)

# XGBOOST
    classifier = xgb.XGBClassifier(
        max_depth=3, n_estimators=300, learning_rate=0.05)
    classifier.fit(df_train.values, ytrain)
    yResult3 = classifier.predict(df_test)

    res3 = np.linalg.norm(yResult3 - ytest) / \
        math.sqrt(len(df_test.values))
    print("Validation score of XGBOOST: ", res3)

    # PLOTTING GRAPHS
    modelName = ["Linear Regression", "Polynomial Regression", "XGBOOST"]
    ylabel = "Transaction Revenue"
    for i in range(3):
        if(i == 0):  # Model - LINEAR
            yPred = yResult
        elif(i == 1):  # Model - POLYNOMIAL
            yPred = yResult2
        else:  # Model - XGBOOST
            yPred = yResult3

        plotGraph(test_plotColumnsList[0], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Visit Number", modelName[i])
        plotGraph(test_plotColumnsList[1], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Bounces", modelName[i])
        plotGraph(test_plotColumnsList[2], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Hits", modelName[i])
        plotGraph(test_plotColumnsList[3], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. New Visits", modelName[i])
        plotGraph(test_plotColumnsList[4], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Page Views", modelName[i])

    yPred = max(yResult, yResult2, yResult3)
    return yPred


# METHOD 1A
# Using only totals columns
def modelTrainingMethod1A(df_train, df_test, ytrain, ytest, train_plotColumnsList, test_plotColumnsList):

    # LINEAR REGRESSSION
    result = LinearRegression().fit(
        df_train[train_plotColumnsList].values, ytrain)
    yResult = result.predict(df_test[test_plotColumnsList].values)
    res = math.log(np.linalg.norm(yResult - ytest) /
                   math.sqrt(len(df_test.values)))
    print("Validation score of Linear Regression: ", res)

    # POLYNOMIAL REGRESSSION
    degree = 4
    poly = PolynomialFeatures(degree=degree)
    poly_x = poly.fit_transform(df_train[train_plotColumnsList].values)
    regressor = LinearRegression()
    regressor.fit(poly_x, ytrain)
    yResult2 = regressor.predict(poly.fit_transform(
        df_test[test_plotColumnsList].values))

    res2 = np.linalg.norm(yResult2 - ytest) / \
        math.sqrt(len(df_test.values))
    print("Validation score of Polynomial Regression: ", res2)

    # XGBOOST
    classifier = xgb.XGBClassifier(
        max_depth=3, n_estimators=300, learning_rate=0.05)
    classifier.fit(
        df_train[train_plotColumnsList].values, ytrain)
    yResult3 = classifier.predict(
        df_test[test_plotColumnsList].values)

    res3 = np.linalg.norm(yResult3 - ytest) / \
        math.sqrt(len(df_test.values))
    print("Validation score of XGBOOST: ", res3)

    # PLOTTING GRAPHS
    modelName = ["Linear Regression", "Polynomial Regression", "XGBOOST"]
    ylabel = "Transaction Revenue"
    for i in range(3):
        if(i == 0):  # Model - LINEAR
            yPred = yResult
        elif(i == 1):  # Model - POLYNOMIAL
            yPred = yResult2
        else:  # Model - XGBOOST
            yPred = yResult3

        plotGraph(test_plotColumnsList[0], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Visit Number", modelName[i])
        plotGraph(test_plotColumnsList[1], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Bounces", modelName[i])
        plotGraph(test_plotColumnsList[2], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Hits", modelName[i])
        plotGraph(test_plotColumnsList[3], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. New Visits", modelName[i])
        plotGraph(test_plotColumnsList[4], ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Page Views", modelName[i])

    yPred = max(yResult, yResult2, yResult3)
    return yPred


def writeBestPredictionToFile(yPred, fullvisitorIDs_test):

    # The following is the format required by KAGGLE
    df_pred = pd.DataFrame()
    df_pred['fullvisitorIDs'] = fullvisitorIDs_test
    df_pred['Transaction Revenue'] = yPred
    df_pred.groupby(['fullvisitorIDs'])
    df_pred.to_csv('submission.csv', encoding='utf-8', index=False)


def main():
    # Train dataset
    filePath = 'method1_train_v2.csv'
    df_train, ytrain, train_plotColumnsList, fullvisitorIDs_train = readDataset(
        filePath)

    # Test dataset
    filePath = 'method1_test_v2.csv'
    df_test, ytest, test_plotColumnsList, fullvisitorIDs_test = readDataset(
        filePath)

    # Training - Method 1
    yPredMethod1 = modelTrainingMethod1(
        df_train, df_test, ytrain, ytest,  train_plotColumnsList, test_plotColumnsList)

    # Training - Method 1A
    yPredMethod1A = modelTrainingMethod1A(df_train, df_test, ytrain, ytest,
                                          train_plotColumnsList, test_plotColumnsList)

    # writing the submission.csv file
    yPred = max(yPredMethod1, yPredMethod1A)
    writeBestPredictionToFile(yPred, fullvisitorIDs_test)


if __name__ == "__main__":
    main()
