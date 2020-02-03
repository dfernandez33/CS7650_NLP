import numpy as np
import math
from operator import itemgetter
# You need to build your own model here instead of using well-built python packages such as sklearn

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """

    def __init__(self):
        pass

    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass

    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZeor(HateSpeechClassifier):
    """Always predict the 0
    """

    def predict(self, X):
        return [0] * len(X)


# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """

    def __init__(self):
        # Add your code here!
        self.class_counts = {}
        self.class_priors = {}
        self.word_counts = {'hatespeech': {}, 'nonhatespeech': {}}

    def fit(self, X, Y):
        # Add your code here!
        num_examples = X.shape[0]
        self.class_counts['hatespeech'] = sum(1 for label in Y if label == 1)
        self.class_counts['nonhatespeech'] = sum(1 for label in Y if label == 0)
        self.class_priors['hatespeech'] = self.class_counts['hatespeech'] / num_examples
        self.class_priors['nonhatespeech'] = self.class_counts['nonhatespeech'] / num_examples

        for row in range(0, num_examples):  # iterate through every row
            y_i = 'hatespeech' if Y[row] == 1 else 'nonhatespeech'
            for index in range(0, X.shape[1]):
                if index not in self.word_counts[y_i]:
                    self.word_counts[y_i][index] = 0.0

                self.word_counts[y_i][index] += X[row][index]

    def predict(self, X):
        # Add your code here!
        predictions = []
        num_examples = X.shape[0]
        num_Ngrams = X.shape[1]
        hatespeech_sum = sum(self.word_counts['hatespeech'].values())
        nonhatespeech_sum = sum(self.word_counts['nonhatespeech'].values())
        for row in range(0, num_examples):
            hatespeech_score = 0
            nonhatespeech_score = 0
            for index in range(0, num_Ngrams):
                if X[row][index] == 0: continue

                # add-1 smoothing applied
                log_w_given_hatespeech = math.log((self.word_counts['hatespeech'][index] + 1) /
                                                  (hatespeech_sum + num_Ngrams))
                log_w_given_nonhatespeech = math.log((self.word_counts['nonhatespeech'][index] + 1) /
                                                     (nonhatespeech_sum + num_Ngrams))

                hatespeech_score += log_w_given_hatespeech
                nonhatespeech_score += log_w_given_nonhatespeech

            hatespeech_score *= self.class_priors['hatespeech']
            nonhatespeech_score *= self.class_priors['nonhatespeech']

            if nonhatespeech_score < hatespeech_score:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def get_ratios(self):
        ratios = []
        hatespeech_sum = sum(self.word_counts['hatespeech'].values())
        nonhatespeech_sum = sum(self.word_counts['nonhatespeech'].values())
        for index in range(0, len(self.word_counts['hatespeech'])):
            w_given_hatespeech = (self.word_counts['hatespeech'][index] + 1) / (hatespeech_sum + len(self.word_counts['hatespeech']))
            w_given_nonhatespeech = (self.word_counts['nonhatespeech'][index] + 1) / (nonhatespeech_sum + len(self.word_counts['hatespeech']))

            ratios.append((w_given_hatespeech / w_given_nonhatespeech, index))

        sorted_ratios = sorted(ratios, key=lambda x: x[0], reverse=True)
        top_10 = sorted_ratios[:10]
        bot_10 = sorted_ratios[-10:]

        return top_10, bot_10




# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """

    def __init__(self):
        # Add your code here!
        self.learning_rate = 5e-5
        self.epochs = 60000
        self.weights = None
        self.regularization_factor = .001

    def fit(self, X, Y):
        # Add your code here!
        self.weights = np.zeros(X.shape[1])

        for epoch in range(0, self.epochs):
            scores = np.dot(X, self.weights)
            predictions = self.sigmoid(scores)

            # weight update with gradient descent
            losses = Y - predictions
            gradients = np.dot(X.T, losses) + (self.regularization_factor * np.linalg.norm(self.weights))

            self.weights += self.learning_rate * gradients

            if epoch % 1000 == 0:
                print("==================================")
                print("Epoch: " + str(epoch))
                print("Log Likelihood: " + str(self.log_likelihood(X, Y)))
                average_loss = sum(losses) / len(losses)
                print("Average Loss: " + str(average_loss))

    def predict(self, X):
        # Add your code here!
        return np.round(self.sigmoid(np.dot(X, self.weights)))

    def sigmoid(self, scores):
        return 1 / (1 + np.exp(-scores))

    # only used for debugging to check if the log likelihood is converging
    def log_likelihood(self, X, Y):
        scores = np.dot(X, self.weights)
        ll = np.sum(Y * scores - np.log(1 + np.exp(scores))) + ((self.regularization_factor / 2) * np.sum(np.square(self.weights)))
        return ll
