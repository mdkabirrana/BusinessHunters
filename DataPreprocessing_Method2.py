import pandas as pd
import json
import numpy as np
from math import log


# Reading train.csv
def readFile(filePath):
    df = pd.read_csv(filePath)
    return df


# dropping unusable/irrelevant/null/constant columns
# Splitting JSON and making columns out of each type of object
def splittingJSON(df):

    df.drop("hits", axis=1, inplace=True)
    df.drop("customDimensions", axis=1, inplace=True)

    df = df.join(pd.read_json(json.dumps(
        list(df['device'].apply(json.loads)))))

    df = df.join(pd.read_json(json.dumps(
        list(df['geoNetwork'].apply(json.loads)))))

    df = df.join(pd.read_json(json.dumps(
        list(df['totals'].apply(json.loads)))))

    df = df.join(pd.read_json(json.dumps(
        list(df['trafficSource'].apply(json.loads)))))
    print('JSON split completed')
    return df


# dropping unusable/irrelevant/null/constant columns
def droppingColumns(df, dropColumns):
    for column in dropColumns:
        if column in df:
            df.drop(column, axis=1, inplace=True)
    print('Not needed columns dropped')
    return df


# One-hot encoding applied to non-integer columns
def oneHotEncoding(df):
    df = pd.get_dummies(df, columns=['channelGrouping', 'browser', 'deviceCategory',
                                     'operatingSystem', 'city', 'continent', 'country', 'region', 'subContinent'])
    print("one-hot encoding / dummies' operation done")
    return df


# convert nan values to 0, an integer
# applying z-normalization
def applyZNormalization(df):

    df = df.replace(np.nan, 0)

    values = list(df.columns.values)

    values.remove('date')
    values.remove('fullVisitorId')
    values.remove('visitId')
    values.remove('visitStartTime')
    values.remove('transactionRevenue')

    for col in values:
        print('Z-normalized ', col)
        temp = (df[col] - df[col].mean())/df[col].std(ddof=0)
        df.drop(col, axis=1, inplace=True)
        df[col] = temp

    return df


# writing to file preprocessed data
def writeFile(df, fileName):
    df.to_csv(fileName, encoding='utf-8', index=False)


def main():
    # Paths of files to be read
    # train_v2 and test_v2
    filePaths = ['train_v2.csv', 'test_v2.csv']

    # File names after processing
    fileToWriteNames = ['method2_train_v2.csv', 'method2_test_v2.csv']

    # loops 2 times because only 2 files
    for i in range(2):
        df_V1 = readFile(filePaths[i])
        df_V2 = splittingJSON(df_V1)

        dropColumns = ['device', 'geoNetwork', 'sessionId', 'totals', 'trafficSource', 'adNetworkType', 'campaignCode', 'gclId', 'adContent', 'isVideoAd', 'page', 'slot', 'keyword', 'source', 'targetingCriteria', 'medium', 'referralPath', 'longitude',
                       'browserVersion', 'browserSize', 'adwordsClickInfo', 'criteriaParameters', 'operatingSystemVersion', 'socialEngagementType', 'networkLocation', 'latitude', 'cityId', 'screenColors',
                       'screenResolution', 'isMobile', 'mobileInputSelector', 'mobileDeviceModel', 'mobileDeviceInfo', 'mobileDeviceMarketingName', 'mobileDeviceBranding', 'flashVersion', 'language', 'networkDomain', 'campaign', 'metro']
        df_V3 = droppingColumns(df_V2, dropColumns)

        df_V4 = oneHotEncoding(df_V3)

        dropColumns = ['not available in demo dataset', '(not set)']
        df_V5 = droppingColumns(df_V4, dropColumns)

        df_V6 = applyZNormalization(df_V5)

        writeFile(df_V6, fileToWriteNames[i])


if __name__ == "__main__":
    main()
