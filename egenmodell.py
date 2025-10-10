import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

N = 1000 # Total population
beta = 0.3 # Infektionshastighet
gamma = 1/7 # Återhämtningshastighet
alpha = 1/5 # Inkubationshastighet
my = 0.005 # Dödlighetshastighet
ny = 3 # Vaccinationshastighet 
theta1 = 0.107 # Minskning av infektionsrisk efter vacc1, 1 - 0.893 = 0.107
theta2 = 0.05 # Minskning av infektionsrisk efter vacc2, 1 - 0.95 = 0.05
delta = 1 #vaccImmune
vacc1_intervall = 1/49 # Hastighet för V1 --> V2
vacc2_intervall = 1/14 # Hastighet för V2 --> VI
immdose1 = 1/14
immdose2 = 1/7
coeff = np.array([N, beta, gamma, alpha, my, ny, theta1, theta2, delta])
t0 = 0
t1 = 120
h = 0.1
tspan = (t0, t1)
y0 = np.array([N - 5, 0, 5, 0, 0, 0, 0, 0]) # S, E, I, R, D, V1, V2, VI
tt = np.arange(t0, t1 + h, h)

def SEIRmodel_ODE(t, y, n, b, g, a, m, ny, t1, t2, d):
    S, E, I, R, D, V1, V2, VI = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    
    # S (Mottagliga) minskar p.g.a infektion och vaccination
    sus = -b * (I/n) * S - ny
    
    # Beräknar infektionstermer
    expS = b * (I/n) * S * d  # S-->E
    expV1 = b * (I/n) * V1 * t1 # V1-->E (genombrottsinfektion)
    expV2 = b * (I/n) * V2 * t2 # V2-->E (genombrottsinfektion)

    # E (Inkubation) ökar p.g.a alla infektioner och minskar p.g.a utveckling till I
    exp = expS + expV1 + expV2 - (a * E)

    # I (Infektion) ökar p.g.a inkubation och minskar p.g.a återhämtning och död
    inf = (a * E) - (g * I) - (m * I)

    # R (Återhämtning) ökar p.g.a återhämtning från I
    rec = g * I

    # D (Dödlighet) ökar p.g.a dödlighet från I
    dea = m * I

    # V1 (Dos 1) ökar p.g.a vaccination och minskar p.g.a infektion och dos 2
    v_dos1 = ny - expV1 - (vacc1_intervall * V1) - immdose1 * V1

    # V2 (Dos 2) ökar p.g.a V1 och minskar p.g.a infektion och immunitet
    v_dos2 = (vacc1_intervall * V1) - (immdose2 * V2) - expV2
    
    # VI ökar p.g.a immunitet från dos 2
    v_imm = immdose1 * V1 + immdose2 * V2

    return np.array([sus, exp, inf, rec, dea, v_dos1, v_dos2, v_imm])

# Beskriver hur varje tillstånd förändras vid varje reaktion
def stochMatrix_SEIR():  
    sMat = np.array([[-1, 1, 0, 0, 0, 0, 0, 0], # S --> E
                     [0, 1, 0, 0, 0, -1, 0, 0], # V1 --> E
                     [0, 1, 0, 0, 0, 0, -1, 0], # V2 --> E
                     [0, -1, 1, 0, 0, 0, 0, 0], # E --> I
                     [0, 0, -1, 1, 0, 0, 0, 0], # I --> R
                     [0, 0, -1, 0, 1, 0, 0, 0], # I --> D
                     [-1, 0, 0, 0, 0, 1, 0, 0], # S --> V1
                     [0, 0, 0, 0, 0, -1, 0, 1], # V1 --> VI
                     [0, 0, 0, 0, 0, -1, 1, 0], # V1 --> V2
                     [0, 0, 0, 0, 0, 0, -1, 1]]) # V2 --> VI 
    return sMat

# Propensitetsfunktion - ger oss sannolikheten att ett fall sker
def SEIR_prop(y, coeff):
    S, E, I, R, D, V1, V2, VI = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    N, beta, gamma, alpha, my, ny, theta1, theta2, delta = coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], coeff[6], coeff[7], coeff[8]

    # Beräkning av stokastisk konstant
    stoch_constant = beta/N

    # Propensitet 1: S --> E
    a1 = stoch_constant * S * I

    # Propensitet 2: V1 --> E (genombrottsinfektion)
    a2 = stoch_constant * V1 * I * theta1

    # Propensitet 3: V2 --> E (genombrottsinfektion)
    a3 = stoch_constant * V2 * I * theta2

    # Propensitet 4: E --> I (Inkubation)
    a4 = alpha * E
    
    # Propensitet 5: I --> R (Återhämtning)
    a5 = gamma * I

    # Propensitet 6: I --> D (Dödlighet)
    a6 = my * I

    # Propensitet 7: S --> V1 (Vaccination dos 1)
    a7 = ny

    #Propensitet 8: V1 --> V2 (Vaccination dos 1)
    a8 = vacc1_intervall * V1

    # Propensitet 9: V1 --> VI (Immunitet)
    a9 = immdose1 * V1

    #Propensitet 9: V2 --> VI (Immunitet)
    a10 =  immdose2 * V2

    return np.array([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])

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
t, X = SSA(SEIR_prop, stochMatrix_SEIR, y0, tspan, coeff)
# Kör determiska modellen med solve_ivp:
sol = solve_ivp(SEIRmodel_ODE, tspan, y0, t_eval=tt, args=(N, beta, gamma, alpha, my, ny, theta1, theta2, delta))

# Plottar båda lösningarna för att jämföra resultaten
plt.figure(1)
plt.plot(t,X[:,0],'b-',label="Mottaglig")
plt.plot(t,X[:,1],'r-', label="Inkubation")
plt.plot(t,X[:,2],'g-', label="Infektion")
plt.plot(t,X[:,3],'y-', label="Återhämtning")
plt.plot(t,X[:,4],'k-', label="Dödlighet")
plt.plot(t,X[:,5],'c-', label="Vaccinerade dos 1")
plt.plot(t,X[:,6],'tab:pink', label="Vaccinerade dos 2")
plt.plot(t,X[:,7],'tab:purple', label="Immun")
plt.title("Egen modell, stokastisk")
plt.legend()

plt.figure(2)
plt.plot(sol.t,sol.y[0],"b-", label="Mottaglig")
plt.plot(sol.t,sol.y[1],"r-", label="Inkubation")
plt.plot(sol.t,sol.y[2],"g-", label="Infektion")
plt.plot(sol.t,sol.y[3],"y-", label="Återhämtning")
plt.plot(sol.t,sol.y[4],"k-", label="Dödlighet")
plt.plot(sol.t,sol.y[5],"c-", label="Vaccinerade dos 1")
plt.plot(sol.t,sol.y[6],"tab:pink", label="Vaccinerade dos 2")
plt.plot(sol.t,sol.y[7],"tab:purple", label="Immun")
plt.legend()
plt.title("Egen modell, deterministisk")
plt.show()