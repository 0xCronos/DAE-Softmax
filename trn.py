#Training DL via mAdam

import pandas     as pd
import numpy      as np
import utility    as ut


# Training miniBatch 
def train_sft_batch(x,y,W,V,S,param):
    costo = []    
    for i in range(numBatch):   
        ...
        ...        
    return(W,V,costo)

# Softmax's training via mAdam
def train_softmax(x,y,param):
    W,V,S    = ut.iniW(...)    
    ...    
    for Iter in range(1,par1[0]):        
        idx   = np.random.permutation(x.shape[1])
        xe,ye = x[:,idx],y[:,idx]         
        W,V,c = train_sft_batch(xe,ye,W,V,param)
        ...
               
    return(W,Costo)    
 
# Training by using miniBatch
def train_dae_batch(x,w1,v,w2,param):
    numBatch = np.int16(np.floor(x.shape[1]/param[0]))    
    cost= [] 
    for i in range(numBatch):                
        ....               
    return(...)

# DAE's Training 
def train_dae(x,param):        
    W,V,S = ut.iniW(...)        
    ....    
    for Iter in range(1,param):        
        xe     = x[:,np.random.permutation(x.shape[1])]                
        ...    = train_dae_batch(xe,...) 
        ....

    return(...) 



#load Data for Training
def load_data_trn():
    ...    
    return(xe,ye)    

# Configuration of the DAE
def load_cnf_dae():      
    par = np.genfromtxt('cnf_dae.csv',delimiter=',')    
    ...
    return(...)



# Beginning ...
def main():
    p_dae,p_sft = load_cnf_dae()           
    xe,ye       = load_data_trn()   
    W,Xr        = train_dae(xe,p_sae)         
    Ws, cost    = train_softmax(Xr,ye,...)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()
