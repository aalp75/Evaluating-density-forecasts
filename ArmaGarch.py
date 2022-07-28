#!/usr/bin/env python
# coding: utf-8

# In[21]:

import numpy as np
import math
from scipy.special import gamma
from scipy.optimize import minimize
import scipy.stats as stats


# We implement a class $ARMA(p_m,q_m)-GARCH(p_v,q_v)$

# # Utilities function

# In[22]:


def gaussian_logpdf(x,loc=0,scale=1):
    return -0.5 *(math.log(2 * math.pi * scale**2) +pow((x-loc)/scale,2))

def student_logpdf(x,nu,loc=0,scale=1):
    return math.log(gamma((nu+1)/2)) - math.log(gamma(nu/2)) -1/2 * math.log( math.pi * (scale**2) * nu )    - (nu+1)/2 * math.log(1+pow(x-loc,2)/((nu*scale**2)))

def generalized_normal_logpdf(x, beta = 1/2,loc=0,scale=1):
    return math.log(beta) - math.log(2*scale*gamma(1/beta)) - pow(np.abs(x-loc)/scale,beta)


# ## Class ARMA-GARCH

# In[23]:


class ArmaGarch:

    #Arma(r,s)-Garch(p,q)

    def __init__(self,pm=0,qm=0,pv=0,qv=0, dist = 'gaussian', sd = 1, df = 6 , gennorm_beta = 1.0 , skewnorm_alpha = 0):

        self.pm = pm
        self.phi = [0] * pm
        self.qm = qm
        self.theta = [0] * qm

        self.pv = pv
        self.alpha = [0] * pv
        self.qv = qv
        self.beta =  [0] * qv

        self.w = 0

        self.loglikelihood = None
        self.AIC = None
        self.BIC = None

        self.dist = dist
        self.df = df
        self.sd = sd
        self.gennorm_beta = gennorm_beta

        self.success = False

        self.skewnorm_alpha = skewnorm_alpha

        self.w_bounds = [[0.001,None]]
        self.alpha_bounds = []
        self.beta_bounds = []
        self.theta_bounds = []
        self.phi_bounds = []
        for i in range(pm):
            self.phi_bounds += [[-0.99,0.99]]
        for i in range(qm):
            self.theta_bounds += [[0,None]]
        for i in range(pv):
            self.alpha_bounds += [[0.01,0.99]]
        for i in range(qv):
            self.beta_bounds += [[0.01,0.99]]


    def arma_filter(self,phi,theta,data):

        n = len(data)
        mu = [0] #set initalize mean with 0
        for i in range(1,n):
            #print(i)
            ar = 0
            for p in range(self.pm):
                if p <=i:
                    ar+= phi[p] * data[i-1-p]
            ma = 0
            for q in range(self.qm):
                if q <= i:
                    ma += theta[q] * (data[i-1-q] - mu[i-1-q])
            mu.append(ar + ma)
        return mu

    def garch_filter(self,w,alpha,beta,data):

        n = len(data)
        sigma2 = [1] #set initalize variance equal to unconditional variance
        for i in range(1,n):
            g = 0
            for p in range(self.pv):
                if p<=i:
                    g+= alpha[p] * data[i-1-p]**2
            arch = 0
            for q in range(self.qv):
                if q <=i:
                    arch += beta[q] * sigma2[i-1-q]
            sigma2.append(w + g + arch)
        return sigma2

    def compute_loglikelihood(self,data):

        n= len(data)

        sigma2 = self.garch_filter(self.w,self.alpha,self.beta,data)
        mu  = self.arma_filter(self.phi,self.theta,data)

        res = 0
        if self.dist=="gaussian":
            for i in range(n):
                if (self.pv>0 or self.qv>0):
                    scale_factor = pow(sigma2[i],1/2) * self.sd
                else:
                    scale_factor = self.sd
                res += gaussian_logpdf(data[i],loc = mu[i], scale = self.sd * scale_factor)
        if self.dist=='student':
            for i in range(n):
                if (self.pv>0 or self.qv>0):
                    scale_factor = pow(sigma2[i]*(self.df-2)/self.df,1/2)
                else:
                    scale_factor = pow((self.df-2)/self.df,1/2)
                res += student_logpdf(data[i], self.df, loc = mu[i], scale = scale_factor)
        if self.dist=='generalize normal':
            for i in range(n):
                if (self.pv>0 or self.qv>0):
                    scale_factor = pow(sigma2[i],1/2)
                else:
                    scale_factor = 1
                res += generalized_normal_logpdf(data[i], beta = self.gennorm_beta, loc = mu[i], scale = scale_factor)
        if self.dist=='skew normal':
            for i in range(n):
                if (self.pv>0 or self.qv>0):
                    scale_factor = pow(sigma2[i],1/2)
                else:
                    scale_factor = 1
                res += stats.skewnorm.logpdf(data[i], self.skewnorm_alpha, loc = mu[i], scale = scale_factor)
        self.loglikelihood = res

    def compute_criteria(self,n):
        if self.w == 0 :
            self.AIC = 2 * (self.pm + self.qm + self.pv + self.qv + 1) - 2 * self.loglikelihood
            self.BIC =  (self.pm + self.qm + self.pv + self.qv + 1) * math.log(n) - 2 * self.loglikelihood
        else:
            self.AIC = 2 * (self.pm + self.qm + self.pv + self.qv + 1) - 2 * self.loglikelihood
            self.BIC =  (self.pm + self.qm + self.pv + self.qv) * math.log(n) - 2 * self.loglikelihood
        if self.dist == 'student':
            self.AIC += 2 * self.df
            self.BIC += math.log(n) * self.df
        if self.dist == 'gaussian' or self.dist == 'generalize normal' or self.dist == 'skew normal':
            self.AIC += 2
            self.BIC += math.log(n)

    def arma_midfs_loglikelihood(self,params,data):

        if self.pm > 0:
            phi = params[:self.pm]
        else:
            phi = []
        if self.qm > 0 :
            theta = params[self.pm:]
        else:
            theta =[]

        for i in range(self.pm):
            if np.abs(phi[i]) >= 1:
                return np.inf

        for i in range(self.qm):
            if theta[i]<=0:
                return np.inf

        n= len(data)

        mu = self.arma_filter(phi,theta,data)

        res = 0
        if self.dist=="gaussian":
            for i in range(n):
                res -= gaussian_logpdf(data[i],loc = mu[i], scale=1)
        if self.dist=='student':
            for i in range(n):
                res -= student_logpdf(data[i], 6 , loc = mu[i], scale = 1)
        if self.dist=='generalize normal':
            for i in range(n):
                res -= generalized_normal_logpdf(data[i], beta=1,loc = mu[i], scale = 1)
        if self.dist == 'skew normal':
            for i in range(n):
                res -= stats.skewnorm.logpdf(data[i], 0, loc = mu[i], scale=1 )
        return res

    def garch_midfs_loglikelihood(self,params,data):


        w = params[0]
        if self.pv > 0:
            alpha = params[1:self.pv+1]
        else:
            alpha = []
        if self.qv > 0 :
            beta = params[self.pv+1:self.pv+self.qv+1]
        else:
            beta =[]

        if self.dist == 'gaussian':
            sd = params[self.pv+self.qv+1]

        if self.dist == 'generalize normal':
            gennorm_beta = params[self.pv+self.qv+1]

        if self.dist == 'skew normal':
            skew_alpha = params[self.pv+self.qv+1]


        for i in range(self.pv):
            if alpha[i]<0:
                return np.inf
        for i in range(self.qv):
            if beta[i]<0:
                return np.inf

        if (w <=0):
            return np.inf

        if sum(alpha) + sum(beta) >= 1 :
            return np.inf

        sigma2 = self.garch_filter(w,alpha,beta,data)

        n= len(data)

        res = 0
        if self.dist=="gaussian":
            for i in range(n):
                scale_factor = pow(sigma2[i],1/2) * sd
                res -= gaussian_logpdf(data[i],loc = 0, scale=scale_factor)
        if self.dist=='student':
            for i in range(n):
                scale_factor = pow(sigma2[i]*(self.df-2)/self.df,1/2)
                res -= student_logpdf(data[i], self.df , loc = 0, scale = scale_factor)
        if self.dist == 'generalize normal':
            for i in range(n):
                scale_factor = pow(sigma2[i],1/2)
                res -= generalized_normal_logpdf(data[i], beta = gennorm_beta, loc = 0, scale=scale_factor)
        if self.dist == 'skew normal':
            for i in range(n):
                scale_factor = pow(sigma2[i],1/2)
                res -= stats.skewnorm.logpdf(data[i], skew_alpha, loc = 0, scale=scale_factor)
        return res

    def fit_arma(self,data):
        phi = []
        theta=[]
        if self.pm > 0 :
            phi = [0.3/self.pm] * self.pm
        if self.qm > 0 :
            theta = [0.3/self.qm] * self.qm
        params_initial = phi + theta
        res = minimize(self.arma_midfs_loglikelihood,x0 =params_initial, args=data , method = 'Nelder-Mead')
        self.phi= res.x[:self.pm]
        self.theta = res.x[self.pm:]
        return


    def fit_garch(self,data):
        w = [0.05]
        alpha = []
        beta=[]
        alpha_sum = 0.1
        for i in range(self.pv):
            alpha.append(2/3*alpha_sum)
            alpha_sum = alpha_sum *1/3
        if (self.pv >0):
            alpha[0] += alpha_sum
        beta_sum = 0.7
        for i in range(self.qv):
            beta.append(2/3*beta_sum)
            beta_sum = beta_sum*1/3
        if self.qv >0:
            beta[0] += beta_sum
        params_initial = []
        if self.dist=='gaussian' :
            params_initial =  w + alpha + beta + [1]
        if self.dist =='student':
            params_initial = w + alpha + beta
        if self.dist == 'generalize normal':
            params_initial = w + alpha + beta + [1/2]
        if self.dist == 'skew normal':
            params_initial = w + alpha + beta + [-1]
        res = minimize(self.garch_midfs_loglikelihood,x0 = params_initial, args=data, method ='Nelder-Mead')
        self.success = res.success
        self.w = res.x[0]
        self.alpha = res.x[1:self.pv+1]
        self.beta = res.x[self.pv+1:]
        if self.dist=='gaussian':
            self.sd =res.x[self.pv+self.qv+1]
        if self.dist == 'generalize normal':
            self.gennorm_beta = res.x[self.pv+self.qv+1]
        if self.dist == 'skew normal':
            self.skewnorm_alpha = res.x[self.pv+self.qv+1]
        return

    def fit(self,data):
        if (self.pm > 0 or self.qm > 0):
            self.fit_arma(data)
        if (self.pv > 0 or self.qv> 0):
            self.fit_garch(data)
        self.compute_loglikelihood(data)
        self.compute_criteria(len(data))

    def show(self):
        print("                 ARMA(%d,%d)-GARCH(%d,%d)" %(self.pm,self.qm,self.pv,self.qv))
        print("================================================")
        if self.dist == "gaussian":
            print("Distribution           Normal")
            print("Standard deviation     %0.2f" %(self.sd))
        if self.dist == "student":
            print("Distribution           Student")
            print("Degree of freedom      %d" %(self.df))
        if self.dist == 'generalize normal':
            print("Distribution          Generalize Normal")
            print("Beta                  %0.2f" %(self.gennorm_beta))
        if self.dist == 'skew normal':
            print("Distribution           Skew normal")
            print("Alpha                  %0.2f" %(self.skewnorm_alpha))
        print("Method                 Maximum Likelihood")
        print("Optimization method    Nelder-Mead")
        print("Sucess                 %s"%(self.success))
        print("Log Likelihood        ", self.loglikelihood)
        print("AIC                   ", self.AIC)
        print("BIC                   ", self.BIC)
        print("================================================\n                       Coeffs\n================================================")
        if (self.pm >0 or self.qm >0):
            print("--------------------Mean Model------------------")
        for i in range(self.pm):
            print("phi[%d]     "%(i), end='')
            print(f"{self.phi[i]:<10}")
        for i in range(self.qm):
            print("theta[%d]   "%(i), end='')
            print(f"{self.theta[i]:<10}")
        if (self.pv >0 or self.qv >0):
            print("-----------------Volatility Model---------------")
        if self.w>0 :
            print("omega      ", end='')
            print(f"{self.w:<10}")
        for i in range(self.pv):
            print("alpha[%d]   "%(i), end='')
            print(f"{self.alpha[i]:<10}")
        for i in range(self.qv):
            print("beta[%d]    "%(i), end='')
            print(f"{self.beta[i]:<10}")
