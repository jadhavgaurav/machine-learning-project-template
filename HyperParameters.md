# Supervised Learning Algorithms & Hyperparameters

## 📌 Introduction
This document provides an exhaustive list of **supervised learning algorithms** (classification & regression), their **hyperparameters** with possible values, and a **sample GridSearchCV tuning process**.

---

## 📚 Table of Contents  
- [Linear Regression](#linear-regression)  
- [Logistic Regression](#logistic-regression)  
- [Decision Tree](#decision-tree)  
- [Random Forest](#random-forest)  
- [Support Vector Machine (SVM & SVR)](#support-vector-machine-svm--svr)  
- [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)  
- [Naive Bayes](#naive-bayes)  
- [Gradient Boosting](#gradient-boosting)  
- [AdaBoost](#adaboost)  
- [XGBoost](#xgboost)  
- [LightGBM](#lightgbm)  
- [CatBoost](#catboost)  
- [Hyperparameter Tuning with GridSearchCV](#hyperparameter-tuning-with-gridsearchcv)  
  - [Best Parameters & Model](#getting-the-best-model-and-parameters)  
  - [Accessing Search Results](#accessing-search-results)  

---

## 📌 Linear Regression  
A simple regression model that predicts a continuous target variable.  
### 🔧 **Hyperparameters & Default Values**  
- `fit_intercept`: Whether to calculate the intercept (`True`, default=`True`)  
- `normalize`: Whether to normalize input features (`True` / `False`, default=`False`)  
- `solver`: Optimization algorithm (`'lbfgs'`, `'sag'`, `'saga'`, default=`'auto'`)  

---

## 📌 Logistic Regression  
Used for binary/multiclass classification.  
### 🔧 **Hyperparameters & Default Values**  
- `C`: Inverse of regularization strength (`float`, default=`1.0`)  
- `penalty`: Regularization (`'l1'`, `'l2'`, `'elasticnet'`, `'none'`, default=`'l2'`)  
- `solver`: Optimization algorithm (`'liblinear'`, `'lbfgs'`, `'newton-cg'`, `'saga'`, default=`'lbfgs'`)  
- `max_iter`: Maximum iterations (`int`, default=`100`)  

---

## 📌 Decision Tree  
A tree-based model used for classification and regression.  
### 🔧 **Hyperparameters & Default Values**  
- `criterion`: Splitting strategy (`'gini'`, `'entropy'` for classification; `'mse'`, `'mae'` for regression, default=`'gini'`)  
- `max_depth`: Maximum tree depth (`int`, default=`None`)  
- `min_samples_split`: Minimum samples needed to split a node (`int` or `float`, default=`2`)  
- `min_samples_leaf`: Minimum samples needed in a leaf node (`int` or `float`, default=`1`)  
- `max_features`: Number of features used for the best split (`'auto'`, `'sqrt'`, `'log2'`, `None`, default=`None`)  

---

## 📌 Random Forest  
An ensemble of multiple decision trees.  
### 🔧 **Hyperparameters & Default Values**  
- `n_estimators`: Number of trees (`int`, default=`100`)  
- `criterion`: Same as Decision Tree (default=`'gini'`)  
- `max_depth`: Maximum tree depth (`int`, default=`None`)  
- `min_samples_split`: Minimum samples needed to split a node (`int` or `float`, default=`2`)  
- `min_samples_leaf`: Minimum samples needed in a leaf node (`int` or `float`, default=`1`)  
- `bootstrap`: Whether to use bootstrap samples (`True`, default=`True`)  
- `oob_score`: Use Out-of-Bag samples for validation (`True` / `False`, default=`False`)  

---

## 📌 Support Vector Machine (SVM & SVR)  
A powerful model for classification and regression.  
### 🔧 **Hyperparameters & Default Values**  
- `C`: Regularization parameter (`float`, default=`1.0`)  
- `kernel`: Kernel type (`'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, default=`'rbf'`)  
- `degree`: Degree for polynomial kernel (`int`, default=`3`)  
- `gamma`: Kernel coefficient (`'scale'`, `'auto'`, `float`, default=`'scale'`)  
- `epsilon`: Epsilon-tube (for SVR) (`float`, default=`0.1`)  

---

## 📌 K-Nearest Neighbors (KNN)  
A distance-based model for classification and regression.  
### 🔧 **Hyperparameters & Default Values**  
- `n_neighbors`: Number of neighbors (`int`, default=`5`)  
- `weights`: Weight function (`'uniform'`, `'distance'`, default=`'uniform'`)  
- `algorithm`: Search algorithm (`'auto'`, `'ball_tree'`, `'kd_tree'`, `'brute'`, default=`'auto'`)  

---

## 📌 Naive Bayes  
Probabilistic classification algorithm.  
### 🔧 **Hyperparameters & Default Values**  
- `var_smoothing`: Smoothing parameter (`float`, default=`1e-9`)  

---

## 📌 Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost  
Boosting technique that builds trees sequentially.  
### 🔧 **Hyperparameters & Default Values**  
- `n_estimators`: Number of boosting stages (`int`, default=`100`)  
- `learning_rate`: Shrinks contribution of each tree (`float`, default=`0.1`)  
- `max_depth`: Maximum depth of trees (`int`, default=`3`)  

---

## 📌 Hyperparameter Tuning with GridSearchCV  
### 📌 Sample Code  
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
```

---

### 📌 Getting the Best Model and Parameters  
#### ✅ **Best Parameters Found**  
```python
print("Best Parameters:", grid_search.best_params_)
```

#### ✅ **Best Model with Optimal Hyperparameters**  
```python
best_model = grid_search.best_estimator_
print(best_model)
```

#### ✅ **Best Cross-Validation Score**  
```python
print("Best CV Score:", grid_search.best_score_)
```

---
## 📌 Conclusion  
✅ Covers essential supervised learning algorithms, hyperparameters, and **GridSearchCV** tuning.  
✅ Use **GridSearchCV, RandomizedSearchCV, or Hyperopt** for optimization.  

---
## 💡 **Need More?**  
🔗 Check out official documentation:  
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/classes.html)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/)  
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)  
- [CatBoost Documentation](https://catboost.ai/docs/)  




