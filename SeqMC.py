import numpy as np
import time as t

#Gaussian with mu=0, sigma = 1
def phi(x):
    return 1/np.sqrt(2*np.pi) * np.exp(- x**2/(2))

#Casewise function defining y in function of x
def y_x(x):
    x = np.array(x)
    c1 = np.argwhere(x<-0.5)
    c2 = np.argwhere((x>=-0.5)&(x<0.5))
    c3 = np.argwhere(x>=0.5)
    y = np.zeros(len(x))
    y[c1] = 1
    y[c2] = 2
    y[c3] = 3
    return y

#takes a value of y and an array of xs
#returns p(y_t|x_t)
def p_y_given_x(y, xs):
    ys = y_x(xs)
    ps = np.where((ys == y), 1, 0)
    return ps

def sequential_MC(y_dat_s, N_sam):
    # --- initialize -----
    sigma = 0.5
    lamb = 0
    alpha = 0.5
    a = 0.85
    Xs = np.random.randn(N_sam) #first sampling from N(0,1)
    Es = []
    Vars = []
    Res_count = 0 #counts how many time I performed resample
    #w_s = p_y_given_x(y_dat_s[0], Xs)#compute weights
    #w_s = w_s / np.sum(w_s) #normalize them
    w_s = np.repeat(1/N_sam, N_sam)
    # --- iterate ------
    for i, y_d in enumerate(y_dat_s[1:]): 
        w_s = p_y_given_x(y_d, Xs)*w_s #update weights
        w_s = w_s / np.sum(w_s) #normalize them
        E = np.sum(Xs*w_s)
        Es.append(E) #expected
        Vars.append(np.sum(w_s*(Xs-E)**2)) #variance
        #check weight degeneracy
        N_eff = 1/np.sum(w_s**2)
        if N_eff < alpha*N_sam:
            Xs = np.random.choice(Xs, p = w_s) #resample
            w_s = np.repeat(1/N_sam, N_sam)
            Res_count += 1
        eps = sigma*np.random.randn(N_sam)
        Xs = a*Xs + lamb + eps #update the sample
    print('End, resample performed %.f times'%Res_count)
    return Es, Vars, Res_count

#similar algorithm but for performin inference on a
def sequential_MC_a(y_dat_s, N_sam):
    # --- initialize -----
    sigma = 0.5
    lamb = 0
    alpha = 0.4
    a_s = np.random.uniform(size = N_sam) #sample a_i from U(0,1)
    Xs = np.random.randn(N_sam) #first sampling from N(0,1)
    Es = []
    Vars = []
    Res_count = 0 #counts how many time I performed resample
    w_s = np.repeat(1/N_sam, N_sam)
    # --- iterate ------
    for i, y_d in enumerate(y_dat_s[1:]): 
        w_s = p_y_given_x(y_d, Xs)*w_s #update weights
        w_s = w_s / np.sum(w_s) #normalize them
        E = np.sum(a_s*w_s)
        Es.append(E) #expected
        Vars.append(np.sum(w_s*(a_s-E)**2)) #variance
        #check weight degeneracy
        N_eff = 1/np.sum(w_s**2)
        if N_eff < alpha*N_sam:
            Xs = np.random.choice(Xs, p = w_s) #resample
            w_s = np.repeat(1/N_sam, N_sam)
            Res_count += 1
        eps = sigma*np.random.randn(N_sam)
        Xs = a_s*Xs + lamb + eps #update the sample, now depends on a_s!!
    print('End, resample performed %.f times'%Res_count)
    return Es, Vars, Res_count