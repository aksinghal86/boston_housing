# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:07:39 2018

@author: asinghal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.learning_curve as curves

from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor
from IPython import get_ipython
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('/Users/asinghal/Documents/Data/housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)
data.head()
data.shape
data.info()

### General stats
min_price = np.min(prices)
max_price = np.max(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)

print("Min price: ${:,.2f}".format(min_price))
print("Max price: ${:,.2f}".format(max_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price: ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

print(prices.describe())
plt.figure(figsize=(9, 8))
sns.distplot(prices, color='g', bins=100, hist_kws={'alpha': 0.4})

for i, col in enumerate(features.columns):
    sns.regplot(data[col], prices)
    plt.show()

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

data.corr()
cols = ['LSTAT', 'PTRATIO', 'RM', 'MEDV']
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, 
                 yticklabels=cols, xticklabels=cols)
plt.show()


 # Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print("Model has a coefficient of determination, R^2, of {:.3f}.".format(score))

X_train, X_test, y_train, y_test = train_test_split(features, prices,
                                                    test_size=0.20,
                                                    random_state=50)

### Produce learning curves for varying training set sizes and maximum depths
def ModelLearning(X, y):
    cv = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2, random_state=0)
    train_sizes = np.linspace(1, X.shape[0]*0.8-1, 9).astype(int)
    fig = plt.figure(figsize=(10, 7))

    for k, depth in enumerate([1, 3, 6, 10]):
        regressor = DecisionTreeRegressor(max_depth=depth)
        sizes, train_scores, test_scores = curves.learning_curve(regressor,
             X, y, cv=cv, train_sizes=train_sizes, scoring='r2')

        train_std = np.std(train_scores, axis=1)
        train_mean = np.mean(train_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color='r', label='Training Score')
        ax.plot(sizes, test_mean, 'o-', color='g', label='Testing Score')
        ax.fill_between(sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.15, color='r')
        ax.fill_between(sizes, test_mean - test_std,
                        test_mean + test_std, alpha=0.15, color='g')

        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
        ax.set_xlim([0, X.shape[0]*0.8])
        ax.set_ylim([-0.05, 1.05])

    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()

ModelLearning(features, prices)


### Model Complexity
def ModelComplexity(X, y):
    cv = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.2, random_state=0)
    max_depth = np.arange(1,11)
    train_scores, test_scores = curves.validation_curve(DecisionTreeRegressor(), X, y,
        param_name="max_depth", param_range=max_depth, cv=cv, scoring='r2')

    train_std = np.std(train_scores, axis=1)
    train_mean = np.mean(train_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(7, 5))
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.plot(max_depth, train_mean, 'o-', color='r', label='Training Score')
    plt.plot(max_depth, test_mean, 'o-', color='g', label='Validation Score')
    plt.fill_between(max_depth, train_mean - train_std,
                     train_mean + train_std, alpha=0.15, color='r')
    plt.fill_between(max_depth, test_mean - test_std,
                     test_mean + test_std, alpha=0.15, color='g')

    plt.legend(loc='lower right')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Score')
    plt.ylim([-0.05, 1.05])
    plt.show()

ModelComplexity(X_train, y_train)


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    params = {'max_depth':[1,2,3,4,5,6,7,8,9,10]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    grid = GridSearchCV(cv=cv_sets, estimator=regressor, param_grid=params, scoring=scoring_fnc)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))


def PredictTrials(X, y, fitter, data):
    """ Performs trials of fitting and predicting data. """

    # Store the predicted prices
    prices = []

    for k in range(10):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size = 0.2, random_state = k)

        # Fit the data
        reg = fitter(X_train, y_train)

        # Make a prediction
        pred = reg.predict([data[1]])[0]
        prices.append(pred)

        # Result
        print("Trial {}: ${:,.2f}".format(k+1, pred))

    # Display price range
    print("\nRange in prices: ${:,.2f}".format(max(prices) - min(prices)))

PredictTrials(features, prices, fit_model, client_data)


