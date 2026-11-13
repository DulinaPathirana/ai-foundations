# import functions from linreg.py
from foundations.linreg import make_data, train

def test_learns_true_params():
    x,y = make_data():
    # more epochs here
    w,b, _ = train(x,y, lr=0.01, epochs=5000)

    #assert = this must be true if nore the test fails
    #abs(w-3)<0.2 means the w is withing 0.2 of the true value 3
    assert abs(w-3)<0.2
    assert abs(b-2)<0.3 # b is a bit noiser to learn therefore looser tolerance