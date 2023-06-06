# My Utility : auxiliars functions

import pandas as pd
import numpy  as np


# Initialize one-wieght    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
    
# STEP 1: Feed-forward of AE
def dae_forward(x,...):
    ...
    return()    


#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))
# STEP 2: Feed-Backward for DAE
def gradW(a,w2):   
    ...
    ...    
    return(...)        

# Update DAE's weight via mAdam
def updW_madam():
        ...    
    return(...v)
# Update Softmax's weight via mAdam
def updW_sft_rmsprop(w,v,gw,mu):
    ...    
    return(w,v)

# Softmax's gradient
def gradW_softmax(x,y,a):        
    ya   = y*np.log(a)
    ...    
    return(gW,Cost)

# Calculate Softmax
def softmax(z):
        exp_z = np.exp(z-np.max(z))
        return(exp_z/exp_z.sum(axis=0,keepdims=True))


# save weights DL and costo of Softmax
def save_w_dl(...):    
    ...
    