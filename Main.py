from tabnanny import NannyNag

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
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn import metrics, linear_model
from sklearn.feature_selection import RFE

from keras.models import Sequential
from keras.layers import Dense

yCols = ['sum milk 305', 'sum fat 305', 'sum prot 305', 'sum Ecm 305']


def rescale_data(data):
    scaled_data = StandardScaler().fit_transform(data)
    return pd.DataFrame(scaled_data)

def seperateXY(data):
    dataY = pd.DataFrame()
    for col in yCols:
        dataY[col] = data[col]
    data = data.drop(columns=['sum milk 305', 'sum fat 305', 'sum prot 305', 'sum Ecm 305'])
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
    Ma=20
    npa = np.zeros(Ma,dtype=np.float32)
    yCols = ['sum milk 305', 'sum fat 305', 'sum prot 305', 'sum Ecm 305']
    badFeature=[]
    for col in yCols:
        theta, j_theta_history1 = gd(X, Y[col], theta, alpha=alpha, M=Ma)
        plot_graph(j_theta_history1, col)
        p = classifierPredict(theta, X)
        print(p)
        print(Y[col])
        print("Train Accuracy:", sum(p ==  Y[col])[0], "%")
        npa += np.asarray(j_theta_history1, dtype=np.float32)
    plot_graph(np.true_divide(npa, 4), "Y")

    print(j_theta_history1)
    plot_graph(j_theta_history1, alpha)

    j_theta_history2 = gd_mini_batch(X,Y['sum milk 305'],theta,alpha=alpha, M=20)
    plot_graph(j_theta_history2, alpha)

    j_theta_history3 = momentum(X,Y['sum milk 305'],theta,alpha=alpha, M=2)
    plot_graph(j_theta_history3, alpha)

def classifierPredict(theta, X):
    """
    take in numpy array of theta and X and predict the class
    """
    predictions = X.dot(theta)

    return predictions > 0

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

    return theta[j] , j_theta_history

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

def runSGD(X_train, X_test, y_train, y_test,dataname):
    regressor = SGDRegressor(loss='epsilon_insensitive',eta0=0.001)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    # plt.show()
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r', lw=2)
    score = regressor.score(X_test, y_test)
    plt.title('SGD - {0}\n Score = {1:.3f} '.format(str(dataname), score))
    plt.xlabel('Actual ')
    plt.ylabel('Predict')
    plt.show(block=False)
    return regressor

def runSimpleLinearRegression(X_train, X_test, y_train, y_test,dataname):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    # plt.show()
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r', lw=2)
    score = regressor.score(X_test, y_test)
    plt.title('Linear Regression - {0}\n Score = {1:.3f} '.format(str(dataname), score))
    plt.xlabel('Actual ')
    plt.ylabel('Predict')
    plt.show(block=False)
    return regressor

def runRidge(X_train, X_test, y_train, y_test,dataname):
    regressor = linear_model.Ridge()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    # plt.show()
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r', lw=2)
    score = regressor.score(X_test, y_test)
    plt.title('Ridge - {0}\n Score = {1:.3f} '.format(str(dataname), score))
    plt.xlabel('Actual ')
    plt.ylabel('Predict')
    plt.show(block=False)
    return regressor

def runLasso(X_train, X_test, y_train, y_test,dataname):
    regressor = linear_model.Lasso(alpha=0.1)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    # plt.show()
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r', lw=2)
    score = regressor.score(X_test, y_test)
    plt.title('Lasso - {0}\n Score = {1:.3f} '.format(str(dataname), score))
    plt.xlabel('Actual ')
    plt.ylabel('Predict')
    plt.show(block=False)
    return regressor

def runElasticNet(X_train, X_test, y_train, y_test,dataname):
    regressor = linear_model.ElasticNet()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    # plt.show()
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], 'r', lw=2)
    score = regressor.score(X_test, y_test)
    plt.title('Elastic Net - {0}\n Score = {1:.3f} '.format(str(dataname), score))
    plt.xlabel('Actual ')
    plt.ylabel('Predict')
    plt.show()
    return regressor

def runNeuralNetwork(X_train, X_test, Y_train, y_test, dataname):
    numberOfFeature = len(X_train.columns)
    NN_Models = []

    X_val, X_test, Y_val, Y_test = train_test_split(X_test, y_test, test_size=0.5)

    NN_model = Sequential()
    epochs = [100,600]
    for epoch in epochs:
        NN_model = Sequential()

        # The Input Layer :
        NN_model.add(Dense(int(numberOfFeature/2), kernel_initializer='normal', input_shape=(numberOfFeature,), activation='relu'))

        # The Hidden Layers :
        NN_model.add(Dense(128, kernel_initializer='normal', activation='relu'))
        NN_model.add(Dense(128, kernel_initializer='normal', activation='relu'))

        # The Output Layer :
        NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

         # Compile the network :
        NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        NN_model.summary()

        hist = NN_model.fit(X_train, Y_train, epochs=epoch, validation_data=(X_val, Y_val), verbose=0)

        NN_Models.append(NN_model)

    min_Score = 1000
    best_Model = None
    for model in NN_Models:
        score = model.evaluate(X_test, Y_test,verbose=0)[0]
        if score < min_Score:
            min_Score = score
            best_Model = model


    y_pred = best_Model.predict(X_test)
    plt.scatter(Y_test, y_pred)
    plt.title('Neural Networks - {0}\n Score = {1:.3f} '.format(str(dataname), 1-min_Score))
    plt.plot([Y_test.min(), Y_test.max()], [y_pred.min(), y_pred.max()], 'r', lw=2)
    plt.xlabel('Actual ')
    plt.ylabel('Predict')
    plt.show()

    return best_Model




def main():
    Saad = pd.read_csv('Oded-File_Farm_56_Calibration_Saad.csv')
    Givat_Chaim = pd.read_csv('Oded-File_Farm_626_Calibration_Givat_Chaim.csv')
    Saad_And_Givat_Chaim = pd.concat([Saad, Givat_Chaim])

    dataTypes = [Saad,Givat_Chaim , Saad_And_Givat_Chaim]
    dataNames = ['Saad','Givat Chaim' , 'Saad And Givat Chaim']

    class model:
        def __init__(self, name, data, X_train, X_test, y_train, y_test,model,model_Name):
            self._name = name
            self._data = data
            self._X_train = X_train
            self._X_test = X_test
            self._y_train = y_train
            self._y_test = y_test
            self._model = model
            self._model_Name = model_Name


    all_models = [[],[],[],[]]
    for data in range(len(dataTypes)):
        X, all_Y , allData= set_data(dataTypes[data].copy())
        bad_features = Recursive_Feature_Elimination(X,all_Y)
        i=0

        for col in yCols:
            all_models[i].append([])
            Y= all_Y[col]
            improved_X = X.drop(labels= bad_features[i], axis = 1)
            X_train, X_test, y_train, y_test = train_test_split(improved_X, Y, test_size=0.3, random_state=0)

            #run Linear Models:
            all_models[i][data].append(model(dataNames[data], dataTypes[data].copy(), X_train, X_test, y_train, y_test,runElasticNet(X_train, X_test, y_train, y_test, dataNames[data]),'Elastic Net'))

            all_models[i][data].append(model(dataNames[data], dataTypes[data].copy(), X_train, X_test, y_train, y_test,runLasso(X_train, X_test, y_train, y_test, dataNames[data]),'Lasso'))

            all_models[i][data].append(model(dataNames[data], dataTypes[data].copy(), X_train, X_test, y_train, y_test,runSGD(X_train, X_test, y_train, y_test, dataNames[data]),'SGD'))

            all_models[i][data].append(model(dataNames[data], dataTypes[data].copy(), X_train, X_test, y_train, y_test,runRidge(X_train, X_test, y_train, y_test, dataNames[data]),'Ridge'))

            all_models[i][data].append(model(dataNames[data], dataTypes[data].copy(), X_train, X_test, y_train, y_test,runSimpleLinearRegression(X_train, X_test, y_train, y_test, dataNames[data]),'Linear Regression'))

            #all_models[i][data].append(model(dataNames[data], dataTypes[data].copy(), X_train, X_test, y_train, y_test,runNeuralNetwork(X_train, X_test, y_train, y_test,dataNames[data]),'Neural Networks'))
            i+=1
    i = 0
    max_Score_Saad = 0
    max_Score_Givat_Chaim = 0
    max_Score_Saad_And_Givat_Chaim =0
    best_Model_Saad = None
    best_Model_Givat_Chaim = None
    best_Model_Saad_And_Givat_Chaim = None

    for model_A in all_models[i]:
        score = 0
        for model_B in model_A:
            if(i==0):
                if(model_B._model_Name == 'Neural Networks'):
                    score = 1- model_B._model.evaluate(model_B._X_test, model_B._y_test,verbose=0)[0]
                else:
                    score = model_B._model.score(model_B._X_test, model_B._y_test)
                if score > max_Score_Saad:
                    max_Score_Saad = score
                    best_Model_Saad = model_B
            elif(i==1):
                if (model_B._model_Name == 'Neural Networks'):
                    score = 1 - model_B._model.evaluate(model_B._X_test, model_B._y_test, verbose=0)[0]
                else:
                    score = model_B._model.score(model_B._X_test, model_B._y_test)
                if score > max_Score_Givat_Chaim:
                    max_Score_Givat_Chaim = score
                    best_Model_Givat_Chaim = model_B
            else:
                if (model_B._model_Name == 'Neural Networks'):
                    score = 1 - model_B._model.evaluate(model_B._X_test, model_B._y_test, verbose=0)[0]
                else:
                    score = model_B._model.score(model_B._X_test, model_B._y_test)
                if score>max_Score_Saad_And_Givat_Chaim:
                    max_Score_Saad_And_Givat_Chaim = score
                    best_Model_Saad_And_Givat_Chaim = model_B
        i += 1

    print('best Saad Model: {}'.format(best_Model_Saad._model_Name))
    print('best Givat Chaim Model: {}'.format(best_Model_Givat_Chaim._model_Name))
    print('best Saad And Givat Chaim Model: {}'.format(best_Model_Saad_And_Givat_Chaim._model_Name))


if __name__ == "__main__":
    main()
