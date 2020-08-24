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


def plotGraph(x, y, yPred, ylabel, title, modelName, whichMethod):
    plt.scatter(x, y,  color='black')

    # whichMethod equals 0, Method 2
    # else, Method 2A
    if(whichMethod == 0):
        plt.plot(x, yPred, color='blue', linewidth=3)
    else:
        plt.plot(np.unique(x), np.poly1d(np.polyfit(
            x, yPred, 1))(np.unique(x)))

    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend([modelName, 'Original Data'], loc='upper left')
    plt.axis([np.amin(x),
              np.amax(x), np.amin(y), np.amax(y)])

    plt.show()
    return


# METHOD 2
# Applied Z_NORMALIZATION during preprocessing
# Applied PCA now
def modelTrainingMethod2(df_train, df_test, ytrain, ytest):
    whichMethod = 0  # NEEDED FOR plotGraph METHOD
    n_components = 4  # max columns for PCA
    # Initalizing infinity values to compare with
    minimumLinear = 51000000000
    pcaLinearBest = 51
    minimumBoost = 51000000000
    pcaBoostBest = 51
    minimumPoly = 51000000000
    pcaPolyBest = 51

    # PCA for 1 - 4 columns
    for l in range(n_components):

        pca = PCA(n_components=l+1)
        PCATrainData = pca.fit(df_train.values).transform(df_train.values)
        PCATestData = pca.fit(df_test.values).transform(df_test.values)

    # LINEAR REGRESSSION
        result = LinearRegression().fit(PCATrainData, ytrain)
        yResult = result.predict(PCATestData)

        res = np.linalg.norm(yResult - ytest) / \
            math.sqrt(len(df_test.values))
        if res < minimumLinear:
            minimumLinear = res
            pcaLinearBest = l+1
            print('Linear Regression Best Result = ' + str(minimumLinear) +
                  '   Min n_comp = ' + str(pcaLinearBest))
            # PLOTTING GRAPHS
            plotGraph(PCATestData, ytest, yResult, "Transaction Revenue",
                      "Transaction Revenue with PCA", "Linear Regression", whichMethod)

    # POLYNOMIAL REGRESSSION
        degree = 4
        poly = PolynomialFeatures(degree=degree)
        poly_x = poly.fit_transform(PCATrainData)

        regressor = LinearRegression()
        regressor.fit(poly_x, ytrain)
        yResult2 = regressor.predict(poly.fit_transform(PCATestData))

        res2 = np.linalg.norm(yResult2 - ytest) / \
            math.sqrt(len(df_test.values))
        if res2 < minimumPoly:
            minimumPoly = res2
            pcaPolyBest = l+1
            print('Poly Regression Best Result = ' + str(minimumPoly) +
                  '   Min n_comp = ' + str(pcaPolyBest))
            # PLOTTING GRAPHS
            plotGraph(PCATestData, ytest, yResult2, "Transaction Revenue",
                      "Transaction Revenue with PCA", "Polynomial Regression", whichMethod)

    # XGBOOST
        classifier = xgb.XGBClassifier(
            max_depth=3, n_estimators=300, learning_rate=0.05)
        classifier.fit(PCATrainData, ytrain)
        yResult3 = classifier.predict(PCATestData)

        res3 = np.linalg.norm(yResult3 - ytest) / \
            math.sqrt(len(df_test.values))
        if res3 < minimumBoost:
            minimumBoost = res3
            pcaBoostBest = l+1
            print('XGB BOOST Best Result = ' + str(minimumBoost) +
                  '   Min n_comp = ' + str(pcaBoostBest))
            # PLOTTING GRAPHS
            plotGraph(PCATestData, ytest, yResult3, "Transaction Revenue",
                      "Transaction Revenue with PCA", "XGBoost", whichMethod)

    yPred = max(yResult, yResult2, yResult3)
    return yPred


# METHOD 2A
# Applied Z_NORMALIZATION during preprocessing
# NO PCA
# Returns best prediction
def modelTrainingMethod2A(df_train, df_test, ytrain, ytest, train_importantColumns, test_importantColumns):

    # LINEAR REGRESSSION
    result = LinearRegression().fit(
        df_train[train_importantColumns].values, ytrain)
    yResult = result.predict(df_test[test_importantColumns].values)
    res = math.log(np.linalg.norm(yResult - ytest) /
                   math.sqrt(len(df_test.values)))
    print('Linear Regression for important columns ' + str(res))

    # POLYNOMIAL REGRESSSION
    degree = 4
    poly = PolynomialFeatures(degree=degree)
    poly_x = poly.fit_transform(df_train[train_importantColumns].values)

    regressor = LinearRegression()
    regressor.fit(poly_x, ytrain)
    yResult2 = regressor.predict(poly.fit_transform(
        df_test[test_importantColumns].values))

    res2 = np.linalg.norm(yResult2 - ytest) / \
        math.sqrt(len(df_test.values))
    print('Polynomial Regression for important columns ' + str(res2))

    # XGBOOST
    classifier = xgb.XGBClassifier(
        max_depth=3, n_estimators=300, learning_rate=0.05)
    classifier.fit(
        df_train[train_importantColumns].values, ytrain)
    yResult3 = classifier.predict(
        df_test[test_importantColumns].values)

    res3 = np.linalg.norm(yResult3 - ytest) / \
        math.sqrt(len(df_test.values))
    print('XGBoost for important columns ' + str(res3))

    whichMethod = 1  # NEEDED FOR plotGraph METHOD

    # Columns to be used for plotting
    xTestVisitNumber = df_test[test_importantColumns].values[:, 0]
    xTestBounces = df_test[test_importantColumns].values[:, 1]
    xTestHits = df_test[test_importantColumns].values[:, 2]
    xTestNewVisits = df_test[test_importantColumns].values[:, 3]
    xTestPageViews = df_test[test_importantColumns].values[:, 4]

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

        plotGraph(xTestVisitNumber, ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Visit Number", modelName[i], whichMethod)
        plotGraph(xTestBounces, ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Bounces", modelName[i], whichMethod)
        plotGraph(xTestHits, ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Hits", modelName[i], whichMethod)
        plotGraph(xTestNewVisits, ytrain, yPred, ylabel,
                  "Transaction Revenue vs. New Visits", modelName[i], whichMethod)
        plotGraph(xTestPageViews, ytrain, yPred, ylabel,
                  "Transaction Revenue vs. Page Views", modelName[i], whichMethod)

    yPred = max(yResult, yResult2, yResult3)
    return yPred


def writeBestPredictionToFile(yPred, fullvisitorIDs_test):

    # The following is the format required by KAGGLE
    log_y_Pred = [math.log(a + 1, 10) for a in yPred]
    df_pred = pd.DataFrame()
    df_pred['fullvisitorIDs'] = fullvisitorIDs_test
    df_pred['Transaction Revenue'] = log_y_Pred
    df_pred.groupby(['fullvisitorIDs'])
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

    # Training with PCA
    yPredMethod2 = modelTrainingMethod2(df_train, df_test, ytrain, ytest)

    # Training WITHOUT PCA
    yPredMethod2A = modelTrainingMethod2A(df_train, df_test, ytrain, ytest,
                                          train_importantColumns, test_importantColumns)

    # writing the submission.csv file
    yPred = max(yPredMethod2, yPredMethod2A)
    writeBestPredictionToFile(yPred, fullvisitorIDs_test)


if __name__ == "__main__":
    main()
