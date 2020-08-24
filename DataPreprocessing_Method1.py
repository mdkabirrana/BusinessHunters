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


# custom encoding applied to non-integer columns
def customEncoding(df):
    df['date'] = df['date'] - min(df['date'])
    df['visitStartTime'] = df['visitStartTime'] - min(df['visitStartTime'])

    group = df['channelGrouping'].unique().tolist()
    df['channelGrouping'] = [group.index(item) + 1
                             for item in df['channelGrouping']]

    group = df['browser'].unique().tolist()
    df['browser'] = [group.index(item) + 1
                     for item in df['browser']]

    group = df['deviceCategory'].unique().tolist()
    df['deviceCategory'] = [group.index(item) + 1
                            for item in df['deviceCategory']]

    group = df['operatingSystem'].unique().tolist()
    df['operatingSystem'] = [group.index(item) + 1
                             for item in df['operatingSystem']]

    group = df['city'].unique().tolist()
    df['city'] = [group.index(item) + 1
                  for item in df['city']]

    group = df['continent'].unique().tolist()
    df['continent'] = [group.index(item) + 1
                       for item in df['continent']]

    group = df['country'].unique().tolist()
    df['country'] = [group.index(item) + 1
                     for item in df['country']]

    group = df['region'].unique().tolist()
    df['region'] = [group.index(item) + 1
                    for item in df['region']]

    group = df['subContinent'].unique().tolist()
    df['subContinent'] = [group.index(item) + 1
                          for item in df['subContinent']]

    df = df.replace(np.nan, 0)
    df = df.replace('', 0)

    return df


# writing to file preprocessed data
def writeFile(df, fileName):
    df.to_csv(fileName, encoding='utf-8', index=False)


def main():
    # Paths of files to be read
    # train_v2 and test_v2
    filePaths = ['train_v2.csv', 'test_v2.csv']

    # File names after processing
    fileToWriteNames = ['method1_train_v2.csv', 'method1_test_v2.csv']

    # loops 2 times because only 2 files
    for i in range(2):
        df_V1 = readFile(filePaths[i])
        df_V2 = splittingJSON(df_V1)

        dropColumns = ['device', 'geoNetwork', 'sessionId', 'totals', 'trafficSource', 'adNetworkType', 'campaignCode', 'gclId', 'adContent', 'isVideoAd', 'page', 'slot', 'keyword', 'source', 'targetingCriteria', 'medium', 'referralPath', 'longitude',
                       'browserVersion', 'browserSize', 'adwordsClickInfo', 'criteriaParameters', 'operatingSystemVersion', 'socialEngagementType', 'networkLocation', 'latitude', 'cityId', 'screenColors',
                       'screenResolution', 'isMobile', 'mobileInputSelector', 'mobileDeviceModel', 'mobileDeviceInfo', 'mobileDeviceMarketingName', 'mobileDeviceBranding', 'flashVersion', 'language', 'networkDomain', 'campaign', 'metro']
        df_V3 = droppingColumns(df_V2, dropColumns)

        df_V4 = customEncoding(df_V3)

        writeFile(df_V4, fileToWriteNames[i])


if __name__ == "__main__":
    main()
