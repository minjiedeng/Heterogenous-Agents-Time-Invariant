"""

Heterogenous agents in a default model
Version: Mar 6, 2019

"""
from __future__ import division
import numpy as np
import random
import quantecon as qe
from numba import jit
import matplotlib.pyplot as plt

class Inequality_Economy:

    """
    In this economy, the government invests in foreign assets in order to smooth
    the consumption of domestic households. Domestic households are heterogenous
    in their incomes. The aggregate endowment is stochastic.
    
    Parameters
    ----------
    beta : float
        Time discounting parameter
    sigma : float
        Risk-aversion parameter
    r : float
        int lending rate
    rho : float
        Persistence in the income process
    eta : float
        Standard deviation of the income process
    theta : float
        Probability of re-entering financial markets in each period
    alpha1,...,alpha5: float
        Income shares of different types of households
    w1,...,w5: float
        Weight on different types of households
    mu: float
        Tax progressivity
    d1, d2: float
        Punishment on aggregate output in present of default
    ny : int
        Number of points in y grid
    nB : int
        Number of points in B grid
    tol : float
        Error tolerance in iteration
    maxit : int
        Maximum number of iterations
    ---------
    Tax for each type household is y_i-lambda* y_i^(1-mu)
    
    """
    def __init__(self, 
            beta = 0.95,      # discount rate
            sigma = 2.,       # risk aversion
            r = 0.04,         # interest rate
            theta = 0.25,     # prob of regaining access
            rho = 0.945,      # persistence in output 
            eta = 0.025,      # st dev of output shock
            d1 = -0.78,       # penalty parameter1
            d2 = 0.85,        # penalty parameter2
            w = 0.2,          # weight on different types
            mu = 0.181,       # tax progressivity
            alpha1 = 0.034,   # income distribution
            alpha2 = 0.085,
            alpha3 = 0.133,
            alpha4 = 0.201,
            alpha5 = 0.547,                
            ny = 20,          # number of points in y grid
            nB = 200,         # number of points in B grid
            tol = 1e-6,       # error tolerance in iteration
            maxit = 10000):

        # Save parameters
        self.beta, self.sigma, self.r = beta, sigma, r
        self.theta,self.rho, self.eta, self.d1, self.d2 = theta, rho, eta, d1, d2
        self.w, self.mu = w, mu
        self.alpha1,self.alpha2,self.alpha3,self.alpha4,self.alpha5=alpha1,alpha2,alpha3,alpha4,alpha5
        self.ny, self.nB = ny, nB

        # Create grids and discretize Markov process
        self.Bgrid = np.linspace(0, 0.6, nB)
        self.mc = qe.markov.tauchen(rho, eta, 3, ny)
        self.ygrid = np.exp(self.mc.state_values)
        self.Py = self.mc.P

        # Output when in default
        self.def_y = self.ygrid-np.maximum(d1*self.ygrid+d2*(self.ygrid**2),0)

        # Allocate memory
        self.Vd = np.zeros(ny)
        self.Vc = np.zeros((ny, nB))
        self.V = np.zeros((ny, nB))
        self.Q = np.ones((ny, nB)) * .95  # Initial guess for prices
        self.default_prob = np.empty((ny, nB))        

        # Compute the value functions, prices, and default prob 
        self.solve(tol=tol, maxit=maxit)

    def solve(self, tol=1e-6, maxit=10000):
        # Iteration Stuff
        it = 0
        dist = 10.

        # Alloc memory to store next iterate of value function
        V_upd = np.zeros((self.ny, self.nB))

        # == Main loop == #
        while dist > tol and maxit > it:

            # Compute expectations for this iteration
            Vs = self.V, self.Vd, self.Vc
            EV, EVd, EVc = (np.dot(self.Py, v) for v in Vs)

            # Run inner loop to update value functions Vc and Vd. 
            _inner_loop(self.alpha1,self.alpha2,self.alpha3,self.alpha4,self.alpha5,
                        self.w,self.mu,self.ygrid, self.def_y, self.Bgrid, self.Vd, self.Vc, 
                        EVc, EVd, EV, self.Q, self.beta, self.theta, self.sigma)
 
            # Update prices
            Vd_compat = np.repeat(self.Vd, self.nB).reshape(self.ny, self.nB)
            default_states = Vd_compat > self.Vc
            self.default_prob[:, :] = np.dot(self.Py, default_states)
            self.Q[:, :] = (1 - self.default_prob)/(1 + self.r)

            # Update main value function and distance
            V_upd[:, :] = np.maximum(self.Vc, Vd_compat)
            dist = np.max(np.abs(V_upd - self.V))
            self.V[:, :] = V_upd[:, :]

            it += 1
            if it%5 == 0:
                print("Running iteration {} with dist of {}".format(it, dist))
  

@jit(nopython=True)
def _inner_loop(alpha1,alpha2,alpha3,alpha4,alpha5,w,mu,ygrid, def_y, Bgrid, Vd, Vc, EVc,
                EVd, EV, qq, beta, theta, sigma):

    ny, nB = len(ygrid), len(Bgrid)
    zero_ind = nB // 2  # Integer division
    for iy in range(ny):
        y = ygrid[iy]   # Pull out current y
        yd = def_y[iy]  # Pull out current y if default
        
        y1 = alpha1 * y
        y2 = alpha2 * y
        y3 = alpha3 * y        
        y4 = alpha4 * y        
        y5 = alpha5 * y
        
        yd1 = alpha1 * yd
        yd2 = alpha2 * yd
        yd3 = alpha3 * yd       
        yd4 = alpha4 * yd        
        yd5 = alpha5 * yd

        denominator_d = yd1**(1-mu)+yd2**(1-mu)+yd3**(1-mu)+yd4**(1-mu)+yd5**(1-mu)
        denominator = y1**(1-mu)+y2**(1-mu)+y3**(1-mu)+y4**(1-mu)+y5**(1-mu)
        cd1 = yd * (yd1)**(1-mu)/denominator_d
        cd2 = yd * (yd2)**(1-mu)/denominator_d
        cd3 = yd * (yd3)**(1-mu)/denominator_d
        cd4 = yd * (yd4)**(1-mu)/denominator_d     
        cd5 = yd * (yd5)**(1-mu)/denominator_d

        # Compute Vd
        Vd[iy] = u(cd1, sigma)*0.2+u(cd2, sigma)*0.2+u(cd3, sigma)*0.2+u(cd4, sigma)*0.2+u(cd5, sigma)*0.2 + \
                 beta * (theta * EVc[iy, zero_ind] + (1 - theta) * EVd[iy])

        # Compute Vc
        for ib in range(nB):
            B = Bgrid[ib] # Pull out current B
            current_max = -1e14
            for ib_next in range(nB):
                c1 = max((y+qq[iy, ib_next]*Bgrid[ib_next]-B)*y1**(1-mu)/denominator, 1e-14)
                c2 = max((y+qq[iy, ib_next]*Bgrid[ib_next]-B)*y2**(1-mu)/denominator, 1e-14)
                c3 = max((y+qq[iy, ib_next]*Bgrid[ib_next]-B)*y3**(1-mu)/denominator, 1e-14)
                c4 = max((y+qq[iy, ib_next]*Bgrid[ib_next]-B)*y4**(1-mu)/denominator, 1e-14)
                c5 = max((y+qq[iy, ib_next]*Bgrid[ib_next]-B)*y5**(1-mu)/denominator, 1e-14)
                m = u(c1, sigma)*0.2+u(c2, sigma)*0.2+u(c3, sigma)*0.2+u(c4, sigma)*0.2+u(c5, sigma)*0.2 +\
                    beta * EV[iy, ib_next]
                if m > current_max:
                    current_max = m
            Vc[iy, ib] = current_max


@jit(nopython=True)
def u(c, sigma):
    return c**(1-sigma)/(1-sigma)

        
if __name__=="__main__":
    se = Inequality_Economy()
    print('Done')
        
