import numpy as np


# =====================Fake_Data========================
# make_data returns 2 arrays of data, x and y
def make_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X= rng.uniform(0, 10, (n, 3)) # n rows, 3 features each
    true_w = np.array([3.0, -1.0, 2.0]) # weights per feature
    true_b= 2

    y= X @ true_w + true_b + rng.normal(0,1,n) # @ is for dotporoduct

    return X,y

# =====================Train_the_model========================
def train(X: np.ndarray, y:np.ndarray, lr: float = 0.5, epochs: int = 500):
    n, d = X.shape # n datapoints, d features
    w = np.zeros(d) # one weight per feature, start at 0
    b= 0.0
    losses=[] 

    for _ in range(epochs):
        y_hat = X @ w + b
        error = y_hat - y
        losses.append(np.mean(error**2))

        dw = (2/n) * (X.T @ error) # gradient for weights
        db = (2/n) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    return w,b,losses

if __name__ == "__main__":
    X, y= make_data()      # generate the data
    # ====================Normalise_Data========================
    mu, sigma = X.mean(axis=0), X.std(axis=0) # mean and std for each feature
    X_norm = (X-mu)/sigma # normalise each feature to have mean 0 and std 1
    w,b, _ = train(X_norm,y) # train the model
    w_real = w/sigma
    b_real = b- np.sum(w*mu/sigma)
    print("w=", np.round(w_real,3), "b=", np.round(b_real,3))
