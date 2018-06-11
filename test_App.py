from App import App
import numpy as np
from rq import Connection

if __name__ == "__main__":

    w = np.random.randn(10,3)
    x = np.random.randn(3,10)

    app1 = App('app1','dot')
    app1.set_arg(w,x)
    with Connection():
        app1.activate(worker=10)
