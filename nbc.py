import pandas as pd
import numpy as np
import utils as util
import split


class NaiveBayesClassifier():
    def __init__(self):
        self.target_attribute = ""
        self.target_values = []
        self.prior_probabilities = {}
        self.conditional_probabilities = {}

    def fit(self, data, target_attribute=""):
        if target_attribute == "":
            self.target_attribute = data.columns[-1]
        else:
            self.target_attribute = target_attribute
        self.target_values = np.sort(data[self.target_attribute].unique())
        self.prior_probabilities = data[self.target_attribute].value_counts().sort_index()
        self.prior_probabilities = self.prior_probabilities / len(data)
        self.prior_probabilities = self.prior_probabilities.to_dict()

        for col in data.columns[:-1]:
            conditionals = []
            for value in self.target_values:
                df = data[data[self.target_attribute] == value]
                count = df[col].value_counts().sort_index()
                probab = self.laplace_correction(count, len(df), np.sort(data[col].unique()))
                conditionals.append(probab.to_dict())

            self.conditional_probabilities[col] = dict(zip(self.target_values, conditionals))

    def laplace_correction(self, nl, n, possible_values):
        for value in possible_values:
            if value not in nl.index:
                nl[value] = 0
        nl = nl.sort_index()
        nl = nl + 1
        n = n + len(possible_values)
        return nl / n

    def predict(self, data):
        predictions = []
        columns = list(data.columns)
        columns.remove(self.target_attribute)
        for index, row in data.iterrows():
            maxProbab = -1
            predLabel = None
            for target in self.target_values:
                probab = self.prior_probabilities[target]
                for col in columns:
                    curprobab = self.conditional_probabilities[col][target].get(row[col], -1)
                    if curprobab == -1:
                        curprobab = min(list(self.conditional_probabilities[col][target].values()))
                    probab = probab * curprobab
                if maxProbab < probab:
                    maxProbab = probab
                    predLabel = target
            predictions.append(predLabel)
        return predictions

    def get_accuracy(self, data):
        if len(data) == 0:
            return 0
        predictions = self.predict(data)
        actual = data[self.target_attribute]
        return (predictions == actual).value_counts().get(True, 0) / len(actual)
