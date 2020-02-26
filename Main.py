import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle
import os

import seaborn as seabornInstance
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
from sklearn.feature_selection import RFE

from keras.models import Sequential
from keras.layers import Dense


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


def Recursive_Feature_Elimination(X, y):
    regressor = LinearRegression()
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

def gradient_decent(X, Y):
    X = np.insert(X.to_numpy(), 0, 1, 1)
    n, m = X.shape
    theta = np.zeros(m)

    alpha = 0.1
    Ma=1000
    npa = np.zeros(Ma,dtype=np.float32)
    yCols = ['sum milk 305', 'sum fat 305', 'sum prot 305', 'sum Ecm 305']
    badFeature=[]
    for col in yCols:
        j_theta_history1 = gd(X, Y[col], theta, alpha=alpha, M=Ma)
        plot_graph(j_theta_history1, col)
        npa += np.asarray(j_theta_history1, dtype=np.float32)
    plot_graph(np.true_divide(npa, 4), "Y")

    print(j_theta_history1)
    plot_graph(j_theta_history1, alpha)

    j_theta_history2 = gd_mini_batch(X,Y['sum milk 305'],theta,alpha=alpha, M=20)
    plot_graph(j_theta_history2, alpha)

    j_theta_history3 = momentum(X,Y['sum milk 305'],theta,alpha=alpha, M=2)
    plot_graph(j_theta_history3, alpha)


def h_theta(x, theta):
    return np.dot(x, theta)
def j_theta(x, y, theta):
    print('j_theta')
    m, n = x.shape
    J = 0
    for i in range(m):
        J += (h_theta(x[i], theta) - y[i]) ** 2
    return J / (2 * m)

def grad_j_theta(x, y, theta):
    print('grad_j_theta')
    grad_j = []
    m, n = x.shape
    for i in range(n):
        element = 0
        for j in range(m):
            element += (h_theta(x[i], theta) - y[i]) * x[j][i]
        element *= (1 / m)
        grad_j.append(element)
    return grad_j

def gd(x, y, theta, alpha=0.1, M=10 ** 3, delta=10 ** -8, epsilon=10 ** -8):
    k = 0
    m, n = x.shape
    j_theta_history = []
    while k < M:
        print(k)

        for j in range(n):
            for i in range(m):
                theta[j] = theta[j] - (1 / m) * alpha * x[i][j] * (h_theta(x[i], theta) -y[i])
        j_theta_history.append(j_theta(x, y, theta))
        k = k + 1

    return j_theta_history

def gd_mini_batch(x, y, theta, alpha=0.1, M=10 ** 3, delta=10 ** -8, epsilon=10 ** -8, N=100):
    print('gd_mini_batch')
    k = 0
    m, n = x.shape
    j_theta_history = []
    while k < M:
        start = (k * N) % m
        end = (((k + 1) * N) - 1) % m
        if end < start:
            end = m - 1
        for j in range(n):
            for i in range(start, end):
                theta[j] = theta[j] - (1 / m) * alpha * x[i][j] * (h_theta(x[i], theta) - y[i])
        j_theta_history.append(j_theta(x, y, theta))
        k = k + 1

    # print(k) #print k here to check if reached M iterations or stopped before
    return j_theta_history

def momentum(x, y, theta, alpha=0.1, alpha_2=0.1, M=10 ** 3, delta=10 ** -8, epsilon=10 ** -8):
    print('momentum')
    k = 0
    m, n = x.shape
    j_theta_history = []
    v = np.zeros(n)
    while k < M:
        for j in range(n):
            for i in range(m):
                theta[j] = theta[j] - (1 / m) * alpha * x[i][j] * (h_theta(x[i], theta) - y[i])
            v = np.multiply(v, 1 - alpha_2) + np.multiply(grad_j_theta(x, y, theta), alpha_2)
        theta = theta - v
        j_theta_history.append(j_theta(x, y, theta))
        k = k + 1

    return j_theta_history

def plot_graph(data,name):
    plt.xlabel('K')
    plt.ylabel('J(theta)')
    plt.title('GD = ' + str(name))
    plt.plot(data)
    # plt.show()
    plt.savefig(str(name)+'.png')


def runLinearRegression(X, Y, allBadFeture, dataname):  # run linear regression for each y
    regressor = SGDRegressor(loss ='huber')
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

def runNeuralNetwork(X, Y, allBadFeture, dataname):

    for i in range(len(Y.columns)):
        y = Y[Y.columns[i]]
        numberOfFeature = len(X.columns) - len(allBadFeture[i])
        dataWithoutBadFeature = X.drop(columns=allBadFeture[i])
        X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(dataWithoutBadFeature, y, test_size=0.3, random_state=0)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

        NN_model = Sequential()

        # The Input Layer :
        NN_model.add(Dense(int(numberOfFeature/2), kernel_initializer='normal', input_shape=(numberOfFeature,), activation='relu'))

        # The Hidden Layers :
        NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

        # The Output Layer :
        NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        # Compile the network :
        NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        NN_model.summary()


        hist = NN_model.fit(X_train, Y_train, epochs=500, batch_size=32, validation_data=(X_val, Y_val),verbose=0)

        # evaluate the model
        _, train_acc = NN_model.evaluate(X_train, Y_train, verbose=0)
        _, test_acc = NN_model.evaluate(X_test, Y_test, verbose=0)

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('Model loss - {} - {}'.format(dataname,Y.columns[i]))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()



def main():
    saadDataSet = pd.read_csv('Oded-File_Farm_56_Calibration_Saad.csv')
    givatChaimDataSet = pd.read_csv('Oded-File_Farm_626_Calibration_Givat_Chaim.csv')
    saadAndGivatChaimDataSet = pd.concat([saadDataSet, givatChaimDataSet])

    saadX, saadY, saadDates = set_data(saadDataSet.copy())
    givatChaimX, givatChaimY, givatChaimDates = set_data(givatChaimDataSet.copy())
    saadAndGivatChaimX, saadAndGivatChaimY, saadAndGivatDates = set_data(saadAndGivatChaimDataSet.copy())



    saadAllYBadFeture = Recursive_Feature_Elimination(saadX, saadY)
    givatChaimAllYBadFeture = Recursive_Feature_Elimination(givatChaimX, givatChaimY)
    saadAndGivatChaimAllYBadFeture = Recursive_Feature_Elimination(saadAndGivatChaimX, saadAndGivatChaimY)

    # runNeuralNetwork(saadX, saadY, saadAllYBadFeture,'Saad')
    # runNeuralNetwork(givatChaimX, givatChaimY, givatChaimAllYBadFeture,'Givat Chaim')
    # runNeuralNetwork(saadAndGivatChaimX, saadAndGivatChaimY, saadAndGivatChaimAllYBadFeture,'Saad and Givat Chaim')

    runLinearRegression(saadX, saadY, saadAllYBadFeture,'Saad')
    runLinearRegression(givatChaimX, givatChaimY, givatChaimAllYBadFeture,'Givat Chaim')
    runLinearRegression(saadAndGivatChaimX, saadAndGivatChaimY, saadAndGivatChaimAllYBadFeture,'Saad and Givat Chaim')

    gradient_decent(saadX,saadY)
    gradient_decent(givatChaimX, givatChaimY)
    gradient_decent(saadAndGivatChaimX, saadAndGivatChaimY)

if __name__ == "__main__":
    main()
