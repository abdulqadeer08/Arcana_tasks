# Week 5 — Tree Models, Ensembles & Evaluation

Topics covered: Decision Trees, Random Forest, XGBoost basics, classification
metrics (precision, recall, F1, AUC), cross-validation and hyperparameter
tuning. The practical work that applies these ideas is in
`classification_model/`.

---

## 1. Decision Trees

A decision tree splits the data into smaller and smaller groups by asking a
series of yes/no questions about the features, for example
`radius_mean <= 15.0?`. Each question is a node, each answer is a branch, and
the process ends at a leaf that holds a class prediction (in our case,
malignant or benign).

**How a split is chosen.** At every node the algorithm looks at every feature
and every possible threshold, and picks the split that makes the two
resulting groups as "pure" as possible — i.e. each group should contain
mostly one class. Purity is measured with:

- **Gini impurity** — the probability of misclassifying a randomly picked
  point if it were labeled according to the class distribution in that node.
  A pure node (all one class) has Gini = 0.
- **Entropy / information gain** — a measure from information theory that
  works the same way; it is 0 for a pure node and highest when the classes
  are split 50/50.

Both measures usually lead to similar trees. Gini is slightly cheaper to
compute, so it's the scikit-learn default.

**Why trees overfit.** If a tree is allowed to grow without limits, it keeps
splitting until every leaf is pure — often a leaf containing a single
training row. That tree has memorized the training set instead of learning a
general pattern, so it performs perfectly on training data and badly on new
data. This is controlled with stopping rules:

- `max_depth` — how many questions deep the tree can go.
- `min_samples_split` / `min_samples_leaf` — a node (or leaf) must contain at
  least this many samples, which prevents splits based on one or two points.
- `max_leaf_nodes` — a direct cap on tree size.

**Strengths:** easy to interpret (you can literally draw the questions it
asks), no need to scale features, handles non-linear relationships.
**Weaknesses:** a single tree has high variance — a small change in the
training data can produce a very different tree. This is the exact problem
Random Forest solves.

---

## 2. Random Forest

A Random Forest trains many decision trees and averages their votes. Two
sources of randomness keep the trees different from one another, which is
what makes the average useful:

1. **Bagging (bootstrap aggregating).** Each tree is trained on a random
   sample of the training rows, drawn *with replacement*, so every tree sees
   a slightly different dataset.
2. **Feature randomness.** At each split, the tree is only allowed to
   consider a random subset of the features (controlled by `max_features`)
   instead of all of them.

Because each tree makes different mistakes, averaging the votes cancels a lot
of that error out — this is what reduces variance compared to a single deep
tree, without needing to restrict any individual tree as heavily.

**Out-of-bag (OOB) score.** Since each tree only trains on ~63% of the rows
(a property of sampling with replacement), the remaining ~37% can be used as
a free validation set for that tree. Averaging this across all trees gives an
OOB score — an estimate of test performance without setting aside a separate
validation split.

**Feature importance.** A Random Forest can report which features reduced
impurity the most across all its trees, which is a quick way to see which
measurements (e.g. `concave_points_worst`) matter most for the prediction.

**Key hyperparameters:** `n_estimators` (number of trees — more trees reduce
variance further but cost more compute, with diminishing returns),
`max_depth`, `min_samples_leaf`, `max_features`.

---

## 3. XGBoost basics

Random Forest builds trees independently and averages them (bagging).
**Gradient boosting**, which XGBoost implements efficiently, builds trees
*sequentially*: each new tree is trained specifically to correct the errors
(residuals) of the trees built before it. The final prediction is a weighted
sum of all the trees, not a simple average.

This means boosted trees are usually shallow (`max_depth` of 3–6 is typical)
because each tree only needs to fix what previous trees got wrong, not model
the whole problem on its own.

**Key hyperparameters:**

- `n_estimators` — how many boosting rounds (trees) to add.
- `learning_rate` (`eta`) — how much each new tree is allowed to correct the
  previous prediction. Lower values need more trees but generalize better.
- `max_depth` — depth of each individual tree.
- `subsample` / `colsample_bytree` — like Random Forest's randomness, these
  make each boosting round use only a fraction of the rows/columns, which
  reduces overfitting.
- `reg_lambda` / `reg_alpha` — L2/L1 regularization terms added directly to
  the loss function, penalizing complex trees.

**Bagging vs boosting, in one line:** bagging (Random Forest) trains trees in
parallel on random subsets to reduce variance; boosting (XGBoost) trains
trees one after another, each fixing the last one's mistakes, to reduce
bias — and controls variance through the regularization terms above.

---

## 4. Classification metrics

Accuracy alone is misleading on an imbalanced dataset. In the breast cancer
data used for the practical work, benign cases outnumber malignant ones
(357 vs 212), so a model that always predicts "benign" would still score
~63% accuracy while being medically useless. The confusion matrix and the
metrics built on it give a much clearer picture.

**Confusion matrix** (for a binary problem, "positive" = malignant here):

|                  | Predicted Positive | Predicted Negative |
|------------------|---------------------|---------------------|
| Actual Positive  | True Positive (TP)  | False Negative (FN) |
| Actual Negative  | False Positive (FP) | True Negative (TN)  |

- **False Negative** — the model said "benign" but the tumor was malignant.
  In a medical context this is the costly mistake: a real case gets missed.
- **False Positive** — the model said "malignant" but it was benign. Costly
  too (unnecessary anxiety/biopsy), but less dangerous than a miss.

**Precision** = TP / (TP + FP). Of everything the model flagged as
malignant, how much actually was. High precision means few false alarms.

**Recall** (a.k.a. sensitivity) = TP / (TP + FN). Of everything that was
actually malignant, how much the model caught. High recall means few missed
cases. In screening problems, recall is usually the metric to prioritize,
because a false negative is worse than a false positive.

**F1 score** = 2 × (Precision × Recall) / (Precision + Recall). The harmonic
mean of precision and recall — useful as a single number when both matter and
the classes are imbalanced (unlike accuracy, F1 doesn't reward a model for
just predicting the majority class).

**ROC curve and AUC.** The ROC curve plots the True Positive Rate (recall)
against the False Positive Rate at every possible decision threshold, instead
of just the default 0.5 cutoff. AUC (Area Under the Curve) summarizes that
curve into one number between 0 and 1:

- AUC = 1.0 → perfect separation between the two classes.
- AUC = 0.5 → the model is no better than random guessing.
- AUC is threshold-independent, so it's a good way to compare models before
  deciding where to set the cutoff for a specific business/medical need.

---

## 5. Cross-validation & hyperparameter tuning

A single train/test split gives one estimate of performance, and that
estimate can be lucky or unlucky depending on which rows ended up in the test
set — especially on a dataset of only a few hundred rows.

**K-fold cross-validation** splits the training data into *k* equal folds.
The model is trained on *k − 1* folds and validated on the remaining one,
repeated *k* times so every fold is used for validation exactly once. The *k*
scores are then averaged. This uses the data more efficiently and gives a
more stable estimate of how the model will perform on unseen data.

**Stratified k-fold** is the version used for classification: each fold
keeps the same class ratio as the full dataset (roughly 63% benign / 37%
malignant in every fold). Without stratification, a fold could end up with
very few malignant examples by chance, which would distort the score.

**Hyperparameter tuning.** Hyperparameters (`max_depth`, `n_estimators`,
`learning_rate`, etc.) are not learned from data the way model weights are —
they have to be searched over. Two standard approaches:

- **Grid search** — tries every combination of a fixed set of values for each
  hyperparameter, and cross-validates each combination. Exhaustive but slow
  once there are many hyperparameters.
- **Randomized search** — samples a fixed number of random combinations
  instead of trying all of them. Usually finds a near-optimal combination in
  a fraction of the time grid search would take.

The important discipline here is that tuning must happen using
cross-validation on the *training* set only. The test set is touched exactly
once, at the very end, to report a final, unbiased score — never used to
choose hyperparameters, or the reported performance stops being a fair
estimate of how the model behaves on new data.
