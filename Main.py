import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle
import os

import seaborn as seabornInstance
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.feature_selection import RFE


def rescale_data(data):
    scaled_data = StandardScaler().fit_transform(data)
    return pd.DataFrame(scaled_data)


def seperateXY(data):
    yCols = ['sum milk 305', 'sum fat 305', 'sum prot 305', 'sum Ecm 305']
    dataY = pd.DataFrame()
    for col in yCols:
        dataY[col] = data[col]
    data = data.drop(columns=['sum milk 305', 'sum fat 305', 'sum prot 305', 'sum Ecm 305'])

    print(data)
    return data, dataY


def set_data(data):
    data = data.loc[(data['DIM'] <= 54)]  # get only the rows with DIM < 54
    data = shuffle(data)
    dfDate = pd.DataFrame()
    for col in ['Date (DD/MM/YYYY)', 'CalvingDate']:
        dfDate[col] = data[col]
    data = data.drop(columns=['Date (DD/MM/YYYY)', 'CalvingDate'])
    all_colums = data.columns
    data = rescale_data(data)
    data = pd.DataFrame(data)
    data.columns = all_colums
    dataX, dataY = seperateXY(data)

    return dataX, dataY, dfDate


def Recursive_Feature_Elimination(regressor, X, y):
    # no of features
    nof_list = np.arange(1, len(X.columns))
    high_score = 0
    # Variable to store the optimum features
    nof = 0
    score_list = []
    yCols = ['sum milk 305', 'sum fat 305', 'sum prot 305', 'sum Ecm 305']
    badFeature = []
    allYBadFeture = []
    for col in yCols:
        y_frame = pd.DataFrame(y[col])
        high_score = 0
        for n in range(len(nof_list)):
            X_train, X_test, y_train, y_test = train_test_split(X, y_frame, test_size=0.3, random_state=0)
            model = LinearRegression()
            rfe = RFE(model, nof_list[n])
            X_train_rfe = rfe.fit_transform(X_train, y_train.values.ravel())
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe, y_train)
            score = model.score(X_test_rfe, y_test)
            score_list.append(score)
            if (score > high_score):
                badFeature = []

                high_score = score
                nof = nof_list[n]
                for i in range(len(X.columns)):
                    if rfe.support_[i] == False:
                        badFeature.append(X.columns[i])
        print("\n\nFor feature {} :".format(col))
        print("Optimum number of features: %d" % nof)
        print("Score with %d features: %f" % (nof, high_score))
        print(badFeature)
        allYBadFeture.append(badFeature)
    return allYBadFeture


def runModel(X, Y, allBadFeture, regressor, dataname):  # run linear regression for each y
    for i in range(len(Y.columns)):
        y = Y[Y.columns[i]]
        plt.figure(figsize=(15, 10))
        plt.title(dataname)
        plt.tight_layout()
        ax = sns.distplot(y)

        dataWithoutBadFeature = X.drop(columns=allBadFeture[i])
        X_train, X_test, y_train, y_test = train_test_split(dataWithoutBadFeature, y, test_size=0.3, random_state=0)

        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)
        y_pred = pd.DataFrame(y_pred)
        plt.show()

        score = regressor.score(X_test, y_test)
        print('Farm: {}, y: {}, score: {}'.format(dataname, Y.columns[i], score))


def main():
    saadDataSet = pd.read_csv('Oded-File_Farm_56_Calibration_Saad.csv')
    givatChaimDataSet = pd.read_csv('Oded-File_Farm_626_Calibration_Givat_Chaim.csv')
    saadAndGivatChaimDataSet = pd.concat([saadDataSet, givatChaimDataSet])

    saadX, saadY, saadDates = set_data(saadDataSet.copy())
    givatChaimX, givatChaimY, givatChaimDates = set_data(givatChaimDataSet.copy())
    saadAndGivatChaimX, saadAndGivatChaimY, saadAndGivatDates = set_data(saadAndGivatChaimDataSet.copy())

    regressor = LinearRegression()

    saadAllYBadFeture = Recursive_Feature_Elimination(regressor, saadX, saadY)
    givatChaimAllYBadFeture = Recursive_Feature_Elimination(regressor, givatChaimX, givatChaimY)
    saadAndGivatChaimAllYBadFeture = Recursive_Feature_Elimination(regressor, saadAndGivatChaimX, saadAndGivatChaimY)

    runModel(saadX, saadY, saadAllYBadFeture, regressor,'Saad')
    runModel(givatChaimX, givatChaimY, givatChaimAllYBadFeture, regressor,'Givat Chaim')
    runModel(saadAndGivatChaimX, saadAndGivatChaimY, saadAndGivatChaimAllYBadFeture, regressor,'Saad and Givat Chaim')


if __name__ == "__main__":
    main()
