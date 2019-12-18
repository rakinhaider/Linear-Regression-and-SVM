import pandas as pd
import numpy as np
import utils as util
from lr import LogisticRegressor
from svm import SVM
from nbc import NaiveBayesClassifier
from discretize import continuousToBinConverter
import sys


def lr(trainingSet, testSet, step_size=0.01, reg_param=0.01, max_iter=500, tol=1e-7):
    model = LogisticRegressor()
    return train_and_test(model, trainingSet, testSet, max_iter=max_iter, step_size=step_size, reg_param=reg_param,tol=tol)


def svm(trainingSet, testSet, step_size=0.5, reg_param=0.01, max_iter=500, tol=1e-7):
    model = SVM()
    trainingSet = trainingSet.copy(deep=True)
    trainingSet[util.target_attribute] = trainingSet[util.target_attribute].replace(to_replace=0, value=-1)
    testSet = testSet.copy(deep=True)
    testSet[util.target_attribute] = testSet[util.target_attribute].replace(to_replace=0, value=-1)
    return train_and_test(model, trainingSet, testSet, step_size=step_size, reg_param=reg_param, max_iter=max_iter, tol=tol)


def nbc(train_set, test_set):
    model = NaiveBayesClassifier()
    model.fit(train_set)
    return model.get_accuracy(train_set)*100, model.get_accuracy(test_set)*100


def train_and_test(model, trainingSet, testSet, step_size=0.01, reg_param=0.01, max_iter=500, tol=1e-7):
    train_X, train_Y = getXY(trainingSet)
    model.fit(train_X, train_Y, max_iter=max_iter, step_size=step_size, reg_param=reg_param, tol=tol)
    training_accuracy = model.get_accuracies(train_X, train_Y)

    test_X, test_Y = getXY(testSet)
    test_accuracy = model.get_accuracies(test_X, test_Y)
    return training_accuracy, test_accuracy


def getXY(data):
    columns = data.columns.tolist()
    Y = data[columns[-1]]
    columns = columns[:-1]
    X = data[columns]
    if not util.final:
        X = pd.DataFrame(data[['age', 'age_o']])
    X['intercept'] = np.ones(len(data))
    return X, Y


if __name__ == "__main__":
    training_data_file = 'trainingSet.csv'
    test_data_file = 'testSet.csv'
    modelIdx = 1

    if len(sys.argv) > 1:
        training_data_file = sys.argv[1]
        test_data_file = sys.argv[2]
        modelIdx = int(sys.argv[3])

    if util.final == True:
        columns, train = util.readFile(training_data_file)
        _, test = util.readFile(test_data_file)
    else:
        columns, train = util.readFile('test_' + training_data_file)
        _, test = util.readFile('test_' + test_data_file)

    if modelIdx == 1:
        training_accuracy, test_accuracy = lr(train, test)
        model_name = 'LR'
    elif modelIdx == 2:
        training_accuracy, test_accuracy = svm(train, test, step_size=0.5, reg_param=0.01)
        model_name = 'SVM'


    print('Training Accuracy {}: {:.2f}'.format(model_name, training_accuracy))
    print('Testing Accuracy {}: {:.2f}'.format(model_name, test_accuracy))
