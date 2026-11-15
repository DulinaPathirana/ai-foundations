import numpy as np

# =====================sigmoid========================
# squash every numbe in to the range 0 to 1
# np.exp(-z) = e^-z
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-z))

# =====================Fake_Data========================
def make_data(n: int = 30, seed: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # hours studied for each student between 0 and 10
    # shape (n,1): n rows, 1 feature
    X= rng.uniform(0,10, (n,1))

    # true rule z= 1.5 * hours - 7.5, 50/50 at 5hrs
    true_w=np.array([1.5])
    true_b= -7.5
    p_true = sigmoid(X@true_w + true_b) # each students chance of passing

    # Flip a weighted coin for each student:
    # rng.uniform(0, 1, n) gives a random number in [0, 1) per student.
    # If it's below their pass probability, they pass (True), otherwise fail (False).
    # .astype(float) turns True/False into 1.0/0.0.
    y= (rng.uniform(0,1,n) < p_true).astype(float)

    return X,y

# =====================Loss_trackingOnly========================
def cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    eps=1e-12  # stops log(0) = -infinity if p hits exactly 0 or 1
    p = np.clip(p, eps, 1-eps) # keep p strictly between 0 and 1
    return -np.mean(y*np.log(p) + (1-y)*np.log(1 - p))

# =====================Train_the_model========================
def train(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 5000):
    n, d = X.shape # n students, d features
    w= np.zeros(d) # start with weight 0
    b = 0.0 # bias 0
    losses = []

    for _ in range(epochs):
        z= X @ w +b                         # linear score for each student
        p = sigmoid (z)                     # turns score into probabilities
        losses.append(cross_entropy(p, y))  # how wrong overall

        error = p-y                         # difference between predicted and actual 
        dw = (1/n) * (X.T @ error)
        db = (1/n) * (np.sum(error))

        w -= lr* dw
        b -= lr* db

    return w,b,losses

# ====================Predict========================
# Turns probabilities into hard 0/1 decisions using the 0.5 threshold.
def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return (sigmoid(X@w+b)>0.5).astype(float)

# ====================Run=========================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, y= make_data() # generate the data
    w, b, losses = train(X,y) # train the model

    acc= np.mean(predict(X,w,b)==y)
    print(f"w= {w[0]:.2f}, b= {b:.2f}, accuracy = {acc:.1%}")

    # plot: dots are the students, curve is the probability learned
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))

    ax1.scatter(X[:,0], y, s=10, alpha=0.5)# X[:, 0] = the hours column
    xs = np.linspace(0,10,200).reshape(-1,1)# smooth range of hours, as (200, 1)
    ax1.plot(xs, sigmoid(xs @ w + b), color="red")      # learned S-curve
    ax1.axhline(0.5, linestyle="--", color="gray")      # decision threshold
    ax1.set_xlabel("hours studied")
    ax1.set_ylabel("P(pass)")
    ax1.set_title("Learned probability")

    ax2.plot(losses)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("cross-entropy")
    ax2.set_title("Loss over training")

    plt.savefig("logreg.png", dpi=120)
    print("Saved plot to logreg.png")