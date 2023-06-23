#Training DL via mAdam
import numpy      as np
import utility    as ut


# Training miniBatch for softmax
def train_sft_batch(ann, X, Y, params):
    minibatch_size = params['sft_minibatch_size']
    amount_of_batches = np.int16(np.floor(X.shape[1] / minibatch_size))
    costs = []

    for n in range(amount_of_batches):
        idx = get_batch_indexes(minibatch_size, n)
        Xb, Yb = X[:, idx], Y[:, idx]
        Yb_pred = ut.sft_forward(ann, Xb, params)
        gradW = ut.gradW_softmax(ann, Yb, params)
        _ = ut.updW_sft_mAdam(ann, gradW, params)
        cost = ut.calculate_sft_cost(Yb, Yb_pred, params)
        costs.append(cost)

    return costs


def create_ann(X, Y):
    WL = ut.iniW(Y.shape[0], X.shape[0])
    V = np.zeros_like(WL)
    S = np.zeros_like(WL)
    W = [WL]
    A = [None] * (len(W) + 1)
    Z = [None] * (len(W) + 1)
    return {'W': W, 'V': V, 'A': A, 'S': S, 'Z': Z, 'layers': len(W)+1}


def train_softmax(X, Y, params):
    ann = create_ann(X, Y)
    mse = []
    for i in range(params['sft_max_iter']):
        idx = np.random.permutation(X.shape[1])
        Xr, Yr = X[:,idx], Y[:,idx]
        costs = train_sft_batch(ann, Xr, Yr, params)
        mse.append(np.mean(costs))

        if i % 10 == 0 and i != 0:
            print(f'Iteration: {i}', mse[i])

    ut.plot_this([mse], 'graphs/softmax/train', ['MSE'], title='Softmax training')
    return(ann['W'][-1], np.array(mse))

    
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


def generate_new_data(dae, X, params):
    dae['W'] = dae['W'][:len(dae['W'])//2]
    dae['layers'] = (len(dae['W']) + 1)
    dae['A'] = [None] *  dae['layers']
    dae['Z'] = [None] * dae['layers'] 
    return ut.dae_forward(dae, X, params)

# DAE's Training 
def train_dae(X, params):
    encoders_nodes = list(params.values())[11:]
    dae = create_dae(encoders_nodes, X.shape[0])
    
    mse = []
    for _ in range(params['dae_max_iter']):
        Xe = X[:, np.random.permutation(X.shape[1])]      
        costs = train_dae_batch(dae, Xe, params)
        mse.append(np.mean(costs))

    X_prime = generate_new_data(dae, X, params)

    ut.plot_this([mse], 'graphs/dae/train', ['MSE'], title="DAE Training")
    
    return dae['W'], X_prime


#load Data for Training
def load_data_trn():
    X = np.loadtxt('X_train.csv', delimiter=',')
    Y = np.loadtxt('Y_train.csv', delimiter=',')
    return X.T, Y.T


# Beginning ...
def main():
    params = ut.load_config()
    Xe, Ye = load_data_trn()
    W, X_prime = train_dae(Xe, params)
    Ws, costs = train_softmax(X_prime, Ye, params)
    
    W.append(Ws)
    ut.save_w_dl(W, costs)


if __name__ == '__main__':
	main()
