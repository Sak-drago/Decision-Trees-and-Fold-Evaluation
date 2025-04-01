import matplotlib.pyplot as plt
import numpy as np

np.random.seed(70)  
num_samples = 100
x_values = np.random.uniform(0, 2 * np.pi, num_samples)
x_values = np.sort(x_values)  
noise = np.random.normal(0, 0.1, size=x_values.shape)
y_values = np.sin(x_values) + noise

# Reshape x to a column vector
X_matrix = x_values.reshape(-1, 1)
print(X_matrix)

def create_polynomial_features(X, degree):
    num_samples = X.shape[0]
    X_poly = np.ones((num_samples, degree + 1))
    for d in range(1, degree + 1):
        X_poly[:, d] = X[:, 0] ** d
    return X_poly

def fit_polynomial_regression(X_poly, y):
    X_transpose = X_poly.transpose()
    # Compute (X^T X)
    XtX = np.dot(X_transpose, X_poly)
    # (X^T X)
    XtX_inv = np.linalg.inv(XtX)
    #(X^T y)
    Xty = np.dot(X_transpose, y)
    # (X^T X)^(-1) (X^T y)
    theta = np.dot(XtX_inv, Xty)
    return theta

def predict_polynomial(X_poly, theta):
    return np.dot(X_poly, theta)

def calculate_mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def perform_cross_validation(X, y, degree, num_folds=5):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  # shuffle indices
    fold_sizes = (num_samples // num_folds) * np.ones(num_folds, dtype=int)
    fold_sizes[: num_samples % num_folds] += 1
    current = 0
    mse_scores = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))
        current = stop
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        X_train_poly = create_polynomial_features(X_train, degree)
        X_test_poly = create_polynomial_features(X_test, degree)
        
        theta = fit_polynomial_regression(X_train_poly, y_train)
        y_pred = predict_polynomial(X_test_poly, theta)
        mse_scores.append(calculate_mean_squared_error(y_test, y_pred))
    
    return np.mean(mse_scores)

# Try polynomial degrees 1 through 4
degrees = [1, 2, 3, 4]
cv_errors = []

print("Cross-Validation MSE for each degree:")
for d in degrees:
    mse_cv = perform_cross_validation(X_matrix, y_values, degree=d, num_folds=5)
    cv_errors.append(mse_cv)
    print(f"Degree {d}: MSE = {mse_cv:.4f}")

optimal_degree = degrees[np.argmin(cv_errors)]
print("\nOptimal polynomial degree chosen:", optimal_degree)
print("\nCV ERROR:", cv_errors)

X_poly_final = create_polynomial_features(X_matrix, optimal_degree)
theta_final = fit_polynomial_regression(X_poly_final, y_values)

x_grid = np.linspace(0, 2 * np.pi, 500).reshape(-1, 1)
X_grid_poly = create_polynomial_features(x_grid, optimal_degree)

plt.figure(figsize=(12, 8))
plt.plot(x_grid, np.sin(x_grid), label=r"True function: $\sin(x)$", color="purple", linewidth=1.5)
plt.scatter(X_matrix, y_values, label="Noisy training points", color="blue", alpha=0.4)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression (Cross-Validation Selected Degree)")
plt.legend()
plt.show()

y_pred_grid = predict_polynomial(X_grid_poly, theta_final)

plt.figure(figsize=(12, 8))
plt.plot(x_grid, np.sin(x_grid), label=r"True function: $\sin(x)$", color="purple", linewidth=1.5)
plt.scatter(X_matrix, y_values, label="Noisy training points", color="blue", alpha=0.4)
plt.plot(x_grid, y_pred_grid, label=f"Polynomial Regression (degree={optimal_degree})", color="red", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Regression (Cross-Validation Selected Degree)")
plt.legend()
plt.show()
