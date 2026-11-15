import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer

from foundations.logreg import predict, sigmoid, train

THRESHOLD : float=0.8
#==========================Load_Real_Data=========================
data = load_breast_cancer()
X_all = data.data # all coloumns
y_all = data.target.astype(float) # 0 = malignant, 1=benign
feature_names = data.feature_names

# ========================Shuffle+split===========================
rng = np.random.default_rng(0)
idx = rng.permutation(len(y_all)) # random order of rows
split = int(0.8* len(y_all))
train_idx, test_idx = idx[:split], idx[split:]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

# ==========================normalise_with_train_stats_only===========================
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train_n = (X_train - mu)/ sigma
X_test_n = (X_test - mu ) / sigma

# =============================train================================
w,b,losses = train(X_train_n, y_train, lr=0.1, epochs=5000)

train_acc = np.mean(predict(X_train_n,w,b, threshold=THRESHOLD)==y_train)
y_pred = predict(X_test_n,w,b, threshold=THRESHOLD)
test_acc = np.mean(y_pred==y_test)
print(f"w= {np.round(w,2)}, b= {b:.2f}")
print(f"train accuracy = {train_acc:.1%}, test accuracy = {test_acc:.1%}")

# =============================plot_data + decision_boundary=============================
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (11,4.5))

# colour for each tumore by its true label

for label, name, colour in [(0, "malignant", "tab:red"), (1, "benign", "tab:blue")]:
    mask = y_train == label
    ax1.scatter(X_train_n[mask,0], X_train_n[mask,1], s=12, alpha=0.6, color=colour, label=name)

# boundary line x2 = -(w1 * x1 + b) / w2, across the plot's x range

x1_line = np.linspace(X_train_n[:,0].min(), X_train_n[:,0].max(), 100)
x2_line = -(w[0]*x1_line+b)/w[1]
ax1.plot(x1_line, x2_line, color="black", label="decision boundary")

ax1.set_xlabel(f"{feature_names[0]} (normalised)")
ax1.set_ylabel(f"{feature_names[1]} (normalised)")
ax1.set_ylim(X_train_n[:, 1].min() - 0.5, X_train_n[:, 1].max() + 0.5)
ax1.legend()
ax1.set_title("Tumours and learned boundary")

ax2.plot(losses)
ax2.set_xlabel("epoch")
ax2.set_ylabel("cross-entropy")
ax2.set_title("Loss over training")

plt.tight_layout()
plt.savefig("logreg_real.png", dpi = 120)
print("saved plot to logreg_real.png")

tp = np.sum((y_test==0) & (y_pred==0))
fn = np.sum((y_test==0) & (y_pred==1))
fp = np.sum((y_test==1) & (y_pred==0))
tn = np.sum((y_test==1) & (y_pred==1))

print(f"tp = {tp}, fn = {fn}, fp = {fp}, tn = {tn}")
print(f"sum = {tp+fn+fp+tn}")

probs = sigmoid(X_test_n @w +b)
fn_mask = (y_test==0) & (y_pred==1)
print (probs[fn_mask])
