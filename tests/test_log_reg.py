import numpy as np
from foundations.logreg import make_data, predict, sigmoid, train


def test_sigmoid_basics():
    # sigmoid(0) must be exactly 0.5, and outputs must stay in (0, 1)
    assert sigmoid(np.array([0.0]))[0] == 0.5
    out = sigmoid(np.array([-100.0, 100.0]))
    assert 0 <= out[0] < 0.01 and 0.99 < out[1] <= 1


def test_accuracy_above_80_percent():
    X, y = make_data()
    w, b, _ = train(X, y)
    acc = np.mean(predict(X, w, b) == y)
    assert acc > 0.8


def test_decision_boundary_near_5_hours():
    # The model predicts 50/50 where z = 0, i.e. hours = -b / w.
    # True boundary is 7.5 / 1.5 = 5 hours.
    X, y = make_data()
    w, b, _ = train(X, y)
    boundary = -b / w[0]
    assert abs(boundary - 5) < 0.75