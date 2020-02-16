import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle

def rescale_data(data):
    scaled_data = StandardScaler().fit_transform(data)
    return pd.DataFrame(scaled_data)


def seperateXY(data):
    yCols = ['sum milk 305', 'sum fat 305', 'sum prot 305', 'sum Ecm 305']
    dataY = pd.DataFrame()
    for col in yCols:
        dataY[col] = data[col]
    data = data.drop(columns=yCols)
    return data,dataY


def set_data(data):
    data = shuffle(data)
    dfDate = pd.DataFrame()
    for col in ['Date (DD/MM/YYYY)','CalvingDate']:
        dfDate[col] = data[col]
    data = data.drop(columns = ['Date (DD/MM/YYYY)','CalvingDate'])
    dataX, dataY = seperateXY(data)
    data = rescale_data(data)

    return dataX,dataY,dfDate


saadDataSet = pd.read_csv('Oded-File_Farm_56_Calibration_Saad.csv')
givatChaimDataSet = pd.read_csv('Oded-File_Farm_626_Calibration_Givat_Chaim.csv')
print(saadDataSet)
print(givatChaimDataSet)

saadX,saadY,saadDates = set_data(saadDataSet)
givatChaimX,givatChaimY,givatChaimDates = set_data(givatChaimDataSet)

print(saadX)
print(saadY)
print(saadDates)
print(givatChaimX)
print(givatChaimY)
print(givatChaimDates)

for y in saadY:
    model = ExtraTreesClassifier()
    model.fit(saadX, saadY[y].values)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=saadX.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

