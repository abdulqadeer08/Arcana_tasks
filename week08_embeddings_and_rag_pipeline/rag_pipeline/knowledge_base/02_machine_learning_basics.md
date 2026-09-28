# Machine Learning Basics

## Types of machine learning
Supervised learning trains a model on labelled examples, where each input has a known target. Regression predicts a continuous number, such as a house price, while classification predicts a category, such as spam or not spam. Unsupervised learning finds structure in unlabelled data, for example clustering customers with k-means. Reinforcement learning trains an agent to take actions that maximise a reward.

## Train, validation and test split
Data is split into a training set used to fit the model, a validation set used to tune hyperparameters and choose between models, and a test set used only once at the end to estimate performance on unseen data. A common split is 70/15/15 or 60/20/20. Using the test set for tuning leaks information and gives an over-optimistic score.

## Linear and logistic regression
Linear regression fits a straight line (or hyperplane) y = w x + b by minimising the mean squared error. It is evaluated with metrics such as MAE, RMSE and R squared. Logistic regression is a classification model: it passes a linear combination of features through the sigmoid function to produce a probability between 0 and 1, and it is trained by minimising binary cross-entropy (log loss).

## Bias-variance tradeoff
Bias is error from overly simple assumptions; a high-bias model underfits and performs poorly on both training and test data. Variance is error from sensitivity to the training data; a high-variance model overfits, scoring well on training data but poorly on new data. Regularisation, more data, and simpler models reduce variance, while more features and more complex models reduce bias.

## Scikit-learn workflow
Scikit-learn models share a consistent API: create an estimator, call fit on training data, then call predict on new data. Preprocessing steps such as StandardScaler and OneHotEncoder can be chained with a model inside a Pipeline, which prevents data leakage because the scaler is fitted only on the training data.
