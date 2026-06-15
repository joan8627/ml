import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat):
    m = len(y)
    y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))

def compute_gradients(X, y, y_hat):
    m = len(y)
    error = y_hat - y
    dw = (X.T @ error) / m
    db = np.mean(error)
    return dw, db

class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = 0.0
        self.losses = []

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)

        for _ in range(self.n_iters):
            z = X @ self.w + self.b
            y_hat = sigmoid(z)

            loss = compute_loss(y, y_hat)
            self.losses.append(loss)

            dw, db = compute_gradients(X, y, y_hat)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_prob(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_prob(X) >= threshold).astype(int)


"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Preprocess: ALWAYS scale features for gradient descent
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train
model = LogisticRegression(lr=0.1, n_iters=1000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
accuracy = np.mean(preds == y_test)
print(f"Accuracy: {accuracy:.4f}")  # typically ~96-98%
"""
