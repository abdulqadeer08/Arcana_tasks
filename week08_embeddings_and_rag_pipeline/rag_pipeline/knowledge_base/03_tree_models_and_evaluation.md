# Tree-Based Models and Model Evaluation

## Decision trees
A decision tree splits the data with a sequence of if-else questions on feature values. At each node it chooses the split that most reduces impurity, measured with Gini impurity or entropy. Deep trees easily overfit, so hyperparameters such as max_depth and min_samples_leaf are used to limit their growth.

## Random forest
A random forest is an ensemble of many decision trees. Each tree is trained on a bootstrap sample of the data (bagging) and considers a random subset of features at each split. The final prediction is the majority vote for classification or the average for regression. Averaging many decorrelated trees reduces variance and makes random forests robust and hard to overfit.

## XGBoost and gradient boosting
Gradient boosting builds trees sequentially: each new tree is trained to correct the errors (residuals) of the trees before it. XGBoost is an optimised gradient boosting library that adds regularisation, handles missing values, and is very fast. Important hyperparameters are n_estimators, learning_rate, and max_depth. A lower learning rate usually needs more trees but generalises better.

## Classification metrics
Accuracy is the fraction of correct predictions, but it is misleading on imbalanced data. Precision is TP / (TP + FP): of the items predicted positive, how many are truly positive. Recall is TP / (TP + FN): of the truly positive items, how many were found. The F1 score is the harmonic mean of precision and recall. ROC-AUC measures how well the model ranks positives above negatives across all thresholds. A confusion matrix shows the counts of true positives, false positives, true negatives and false negatives.

## Cross-validation and tuning
K-fold cross-validation splits the training data into k folds, trains on k-1 folds and validates on the remaining fold, repeating k times and averaging the scores. Stratified k-fold keeps the class proportions the same in every fold. GridSearchCV tries every combination of hyperparameters with cross-validation, while RandomizedSearchCV samples a fixed number of combinations and is faster for large search spaces.
