import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from numpy.linalg import svd

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['log_price'])
    y = df['log_price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def linear_regression_normal_equation(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

def linear_regression_svd(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    U, sigma, Vt = svd(X_b, full_matrices=False)
    sigma_inv = np.diag(1 / sigma)
    theta_svd = Vt.T.dot(sigma_inv).dot(U.T).dot(y)
    return theta_svd

def polynomial_regression(X_train, X_test, y_train, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    return model, X_test_poly

def batch_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])
    m = len(X_b)
    for _ in range(iterations):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= learning_rate * gradients
    return theta

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def train_sklearn_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_sgd_regressor(X_train, y_train):
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train, y_train, alpha=0.1):
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model
