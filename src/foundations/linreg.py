# =======================================imports=======================================
import numpy as np
import matplotlib.pyplot as plt
# =======================================end_imports=======================================

# =====================Fake_Data========================
# make_data returns 2 arrays of data, x and y
def make_data(n: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # random number generator. Fixed seed therefore the same numbers every time
    rng=np.random.default_rng(seed)

    x=rng.uniform(0, 10, n) # n number of random numbers between 0 and 10

    # true relationship is y=3x+2+noise
    y=3*x+2+rng.normal(0, 1, n)

    return x, y # send both arrays back to the caller

# ====================Train_the_model========================
# lr is the learning rate
# epochs the is the number of repeititions of the training process
def train(x: np.ndarray, y:np.ndarray, lr: float = 0.01, epochs: int = 1000):
    w, b= 0, 0 # start the model at 0,0 
    n= len(x) # number of data points
    losses=[] # list to hold the loss values for each epoch

    for _ in range(epochs):
        # prediction for every point 
        y_hat = w*x + b # predicted values

        # error for every point of y
        error = y_hat - y # difference between predicted and actual values

        # mean squared error loss such that negatives don't cancel out positives
        # .append adds the value to the end of the list
        losses.append(np.mean(error**2))

        # gradient, from the formulas
        # dL/dw = 2/n * sum(error * x)
        # dL/db = 2/n * sum(error)
        # error * x multiplied them elementwise, then sum adds them all together
        dw = (2/n) * np.sum(error * x)
        db = (2/n) * np.sum(error)

        # gradient points uphill, so we need to go in the opposite direction to minimize the loss
        # "w -=" is shorthand for "w= w - ..."
        w -= lr * dw
        b -= lr * db

    return w,b,losses # final parameters + loss history

# ====================Run_It========================
if __name__ == "__main__":
    x, y = make_data()      # generate the data
    w,b,losses = train(x,y) # train the model

    # f-strin: {w:.3f} inserts w rounded to 3 decimal places
    print(f"w: {w:.3f}. b={b:.3f}")

    # make figure with 2 plots
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4)) # fig size is the width, height in inches

    # left plot the data as dots and the learned line in red
    ax1.scatter(x,y, s=10) #s is the dot size
    ax1.plot(x, w*x+b, color="red") # learned line
    ax1.set_title("Fit")

    # right plot, loss vs epoch
    ax2.plot(losses)
    ax2.set_title("Loss over Training")

    plt.savefig("fit.png", dpi=120)
    print("Saved plot to fit.png")