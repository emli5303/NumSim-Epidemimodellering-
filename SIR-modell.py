import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

N = 1000
beta = 0.7
gamma = 1/7
t0 = 0
t1 = 120
h = 0.1
tspan = (t0, t1)
tt = np.arange(t0, t1 + h, h)
y0 = np.array([N - 5, 5, 0])

def SIRmodel_ODE():