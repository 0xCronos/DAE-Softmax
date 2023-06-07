#Training DL via mAdam
import numpy      as np
import utility    as ut


# # Training miniBatch 
# def train_sft_batch(x,y,W,V,S,param):
#     costo = []    
#     for i in range(numBatch):   
#         ...
#         ...        
#     return(W,V,costo)


# # Softmax's training via mAdam
# def train_softmax(x,y,param):
#     W,V,S    = ut.iniW(...)    
#     ...    
#     for Iter in range(1,par1[0]):        
#         idx   = np.random.permutation(x.shape[1])
#         xe,ye = x[:,idx],y[:,idx]         
#         W,V,c = train_sft_batch(xe,ye,W,V,param)
#         ...
#     return(W,Costo)    
 
    
def get_batch_indexes(minibatch_size, n):
    start_index = n * minibatch_size
    end_index = start_index + minibatch_size
    return np.arange(start_index, end_index).astype(int)


# Training by using miniBatch
def train_dae_batch(dae, X, params):
    minibatch_size = params['dae_minibatch_size']
    amount_of_batches = np.int16(np.floor(X.shape[1] / minibatch_size))

    costs = []
    for n in range(amount_of_batches):
        idx = get_batch_indexes(minibatch_size, n)
        Xb = X[:, idx]
        
        X_prime = ut.dae_forward(dae, Xb, params)
        gradW = ut.dae_gradW(dae, params)
        _ = ut.dae_updW_madam(dae, gradW, params)
        
        cost = (np.sum(np.sum((X_prime - Xb) ** 2), axis=0)) / (2 * minibatch_size)
        costs.append(cost)
    
    return costs


# Create deep autoencoder
def create_dae(encoders_nodes, features):
    layers = (len(encoders_nodes) * 2) + 1
    W = [None] * (layers - 1)
    V = [None] * (layers - 1)
    S = [None] * (layers - 1)
    A = [None] * layers
    Z = [None] * layers

    for i, nodes in enumerate(encoders_nodes):
        if i == 0:
            W[i] = ut.iniW(nodes, features) # encoder
        else:
            W[i] = ut.iniW(nodes, W[i-1].shape[0]) # encoders

        W[len(W)-(i+1)] = ut.iniW(*(W[i].T.shape)) # decoders
   
    for i in range(len(W)):
        V[i] = np.zeros_like(W[i])
        S[i] = np.zeros_like(W[i])
    
    return { 'W': W, 'V': V, 'S': S, 'A': A, 'Z': Z, 'layers': layers}


# DAE's Training 
def train_dae(X, params):
    encoders_nodes = list(params.values())[11:]
    dae = create_dae(encoders_nodes, X.shape[0])
    
    mse = []
    for _ in range(params['dae_max_iter']):
        Xe = X[:, np.random.permutation(X.shape[1])]      
        costs = train_dae_batch(dae, Xe, params)
        mse.append(np.mean(costs))
    
    ut.plot_this(
        [mse], 
        'graphs/dae/all.png', 
        ['Cost evolution'],
        title="DAE Training minimization")
    
    return dae['W'][:len(dae['W']) // 2], costs


#load Data for Training
def load_data_trn():
    X = np.loadtxt('X_train.csv', delimiter=',')
    Y = np.loadtxt('Y_train.csv', delimiter=',')
    return X.T, Y.T


# Beginning ...
def main():
    params = ut.load_config()
    Xe, Ye = load_data_trn()
    W, Xr = train_dae(Xe, params)
    #Ws, cost = train_softmax(Xr, Ye)
    #ut.save_w_dl(W, Ws, cost)
       
if __name__ == '__main__':
	main()
