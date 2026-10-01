# Spaceship Titanic

Binary classification on the Kaggle **Spaceship Titanic** dataset: predict whether a passenger was `Transported` to another dimension. Built as Business Challenge #2 during my MSc in Business Analytics (Hult).

## Approach

1. **Exploration and cleaning**: missing-value imputation (most frequent class, "unknown" class, and continuous-column strategies) and correlation analysis of features such as `Deck`, `Destination`, `CryoSleep` and onboard spending.
2. **Feature selection**: iterative logistic regression (statsmodels) to keep statistically significant variables.
3. **Model comparison**: Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting and Ridge Classifier, with hyperparameter tuning via `RandomizedSearchCV`.
4. **Submission**: predictions exported to `submission.csv`.

## Results (holdout set, from the notebook output)

| Model | Train accuracy | Test accuracy | AUC |
|---|---|---|---|
| Random Forest (tuned) | 0.831 | **0.801** | 0.801 |
| Gradient Boosting | 0.954 | 0.793 | 0.793 |

Random Forest was preferred for its much smaller train/test gap (0.03 vs 0.16).

## Stack

Python, pandas, scikit-learn, statsmodels, matplotlib, seaborn
