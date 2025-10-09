import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

N = 1000 # Total population
beta = 0.3 # Infektionshastighet
gamma = 1/7 # Återhämtningshastighet
t0 = 0
t1 = 120
h = 0.1
tspan = (t0, t1)
tt = np.arange(t0, t1 + h, h)
y0 = np.array([N - 5, 5, 0]) # S, I, R

def SIRmodel_ODE(t, y, n, b, g):
    S, I, R = y[0], y[1], y[2]

    # S (Mottagliga) minskar p.g.a infektion
    sus = -beta * (I/N) * S
    
    # I (Infektion) ökar p.g.a inkubation och minskar p.g.a återhämtning
    inf = beta * (I/N) * S - (gamma * I)

    # R (Återhämtning) ökar p.g.a återhämtning från I
    rec = gamma * I

    return np.array([sus, inf, rec])

sol = solve_ivp(SIRmodel_ODE, tspan, y0, t_eval=tt, args=(N, beta, gamma))

plt.plot(sol.t, sol.y[0], label="Mottaglig")
plt.plot(sol.t, sol.y[1], label="Infektion")
plt.plot(sol.t, sol.y[2], label="Återhämtning")
plt.title('SIR-modell')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.show()
