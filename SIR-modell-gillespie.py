import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

N = 1000 # Total population
beta = 0.3 # Infektionshastighet
gamma = 1/7 # Återhämtningshastighet
coeff = np.array([N, beta, gamma])
t0 = 0
t1 = 120
h = 0.1
tspan = (t0, t1)
tt = np.arange(t0, t1 + h, h)
y0 = np.array([N - 5, 5, 0]) # S, I, R


def SIRmodel_ODE(t, y, n, b, g):
    S, I, R = y[0], y[1], y[2]

    # S (Mottagliga) minskar p.g.a infektion
    sus = -b * (I/n) * S

    # I (Infektion) ökar p.g.a inkubation och minskar p.g.a återhämtning
    inf = b * (I/n) * S - (g * I)

    # R (Återhämtning) ökar p.g.a återhämtning från I
    rec = g * I

    return np.array([sus, inf, rec])


def stochMatrix_SIR():
    sMat = np.array([[-1, 1, 0],  # S --> E
                     [0, -1, 1]]) # I --> R
    return sMat

# Propensitetsfunktion - ger oss sannolikheten att ett fall sker
def SIR_prop(y, coeff):
    S, I = y[0], y[1]
    N, beta, gamma = coeff[0], coeff[1], coeff[2]

    # Beräkning av stokastisk konstant
    stoch_constant = beta/N

    # Propensitet 1: Total intensitet för infektion (S --> E)
    a1 = stoch_constant * S * I
    
    # Propensitet 2: Total intesitet för återhämtning (I --> R)
    a2 = gamma * I

    return np.array([a1, a2])

# Gillespie algoritm:
def SSA(prop, stoch, X0, tspan, coeff):
    # prop  - propensities
    # stoch - stiochiometry vector
    # Initial state vector
    tvec = np.zeros(1)
    tvec[0] = tspan[0]
    Xarr = np.zeros([1,len(X0)])
    Xarr[0,:] = X0
    t = tvec[0]
    X = X0
    sMat = stoch()
    while t<tspan[1]:
        r1, r2 = np.random.uniform(0, 1, size=2)  # Find two random numbers on uniform distr.
        re = prop(X,coeff)
        cre = np.cumsum(re)
        a0 = cre[-1]
        if a0 < 1e-12:
            break

        tau = np.random.exponential(scale=1/a0) # Random number exponential distribution
        cre = cre/a0
        r = 0
        while cre[r] < r2:
            r += 1
        
        t += tau
        # if new time is larger than final time, skip last calculation
        if t > tspan[1]:
            break
        
        tvec = np.append(tvec,t)
        X = X + sMat[r,:]
        Xarr = np.vstack([Xarr,X])

     # If iterations stopped before final time, add final time and no change
    if tvec[-1] < tspan[1]:
        tvec=np.append(tvec,tspan[1])
        Xarr=np.vstack([Xarr,X])          
            
    return tvec, Xarr

# Kör stokastiska modellen med Gillespie algoritmen:
t, X = SSA(SIR_prop, stochMatrix_SIR, y0, tspan, coeff)
# Kör determiska modellen med solve_ivp:
sol = solve_ivp(SIRmodel_ODE, tspan, y0, t_eval=tt, args=(N, beta, gamma))


# Plottar båda lösningarna för att jämföra resultaten
plt.figure(1)
plt.plot(t,X[:,0],'b-',label="Mottaglig")
plt.plot(t,X[:,1],'r-', label="Infektion")
plt.plot(t,X[:,2],'y-', label="Återhämtning")
plt.title("SIR-modell, stokastisk")
plt.legend()

plt.figure(2)
plt.plot(sol.t,sol.y[0], "b-", label="Mottaglig")
plt.plot(sol.t,sol.y[1],"r-", label="Infektion")
plt.plot(sol.t,sol.y[2],"y-", label="Infektion")
plt.legend()
plt.title("SIR-modell, deterministisk")
plt.show()

