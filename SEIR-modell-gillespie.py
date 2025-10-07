import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

N = 1000
beta = 0.3
gamma = 1/7
alpha = 1/5
coeff = np.array([N, beta, gamma, alpha])
t0 = 0
t1 = 120
h = 0.1
tspan = (t0, t1)
y0 = np.array([N - 5, 0, 5, 0])
tt = np.arange(t0, t1 + h, h)

def SEIRmodel_ODE(t, y, n, b, g, a):
    S, E, I, R = y[0], y[1], y[2], y[3]

    sus = -b * (I/n) * S

    inf = (a * E) - (g * I)
    
    exp = b * (I/n) * S - (a * E)

    rec = g * I

    return np.array([sus, exp, inf, rec])


def stochMatrix_SEIR():  
    sMat = np.array([[-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
    return sMat

def SEIR_prop(y, coeff):
    S, E, I, R = y[0], y[1], y[2], y[3]
    N, beta, gamma, alpha = coeff[0], coeff[1], coeff[2], coeff[3]

    # Beräkning av stokastisk konstant
    stoch_constant = beta/N

    # Propensitet 1: Total intensitet för infektion (S --> E)
    a1 = stoch_constant * S * I
    
    # Propensitet 2: Total intensitet för inkubation (E --> I)
    a2 = alpha * E
    
    # Propensitet 3: Total intesitet för återhämtning (I --> R)
    a3 = gamma * I

    return np.array([a1, a2, a3])


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

t, X = SSA(SEIR_prop, stochMatrix_SEIR, y0, tspan, coeff)
sol = solve_ivp(SEIRmodel_ODE, tspan, y0, t_eval=tt, args=(N, beta, gamma, alpha))


# plottar båda lösningarna för att jämföra resultaten
plt.figure(1)
plt.plot(t,X[:,0],'b-',label="Mottaglig")
plt.plot(t,X[:,1],'r-', label="Inkubation")
plt.plot(t,X[:,2],'g-', label="Infektion")
plt.plot(t,X[:,3],'y-', label="Återhämtning")
#ax1 = plt.gca()
#xmin, xmax, ymin, ymax = ax1.axis()
# ax1.set(xlim=(0,xmax), ylim=(0, ymax))
plt.title("Mottaglig/inkubation/infektion/återhämtning, stokastisk")
plt.legend()

plt.figure(2)
plt.plot(sol.t,sol.y[0],"b-", label="Mottaglig")
plt.plot(sol.t,sol.y[1],"r-", label="Inkubation")
plt.plot(sol.t,sol.y[2],"g-", label="Infektion")
plt.plot(sol.t,sol.y[3],"y-", label="Återhämtning")
#ax2 = plt.gca()
# xmin, xmax, ymin, ymax = ax2.axis()
#ax2.set(xlim=(0, xmax), ylim=(0, ymax))
plt.legend()
plt.title("Mottaglig/inkubation/infektio/återhämtning, deterministisk")
plt.show()

