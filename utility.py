# My Utility : auxiliars functions
import numpy  as np


# Configurations of autoencoders and softmax as dict 
def load_config():
    cnf_dae = np.genfromtxt('cnf_dae.csv', delimiter=',')    
    cnf_sft = np.genfromtxt('cnf_softmax.csv', delimiter=',')    
    params = {
        'n_classes' : int(cnf_dae[0]),
        'n_frames' : int(cnf_dae[1]),
        'frame_size' : int(cnf_dae[2]),
        'trn_percentage' : float(cnf_dae[3]),
        'dae_enc_act_func' : int(cnf_dae[4]),
        'dae_max_iter' : int(cnf_dae[5]),
        'dae_minibatch_size' : int(cnf_dae[6]),
        'dae_learning_rate' : float(cnf_dae[7]),
        'sft_max_iter': int(cnf_sft[0]),
        'sft_learning_rate': float(cnf_sft[1]),
        'sft_minibatch_size': int(cnf_sft[2]),
    }
    
    for l, nodes in enumerate(cnf_dae[8:]):
        params[f'encoder{l+1}_nodes'] = int(nodes)
        
    return params


def sort_data_random(X, Y, by_column=False):
    if by_column:
        random_indexes = np.random.permutation(X.shape[1])
        random_X = X[:, random_indexes]
        random_Y = Y[:, random_indexes]
    else:
        random_indexes = np.random.permutation(X.shape[0])
        random_X = X[random_indexes, :]
        random_Y = Y[random_indexes, :]

    return random_X, random_Y


# Normalize a list of variables (dataset)
def normalize_data(X):
    for i in range(len(X)):
        X[i] = normalize_var(X[i])
    return X


# Normalize a variable
def normalize_var(var, a=0.01, b=0.99):
    x_min = np.min(var)
    x_max = np.max(var)
    for i in range(len(var)):
        var[i] = (var[i] - x_min) / (x_max - x_min) * (b - a)  + a
    return var


# Activation function
def act_function(Z, activation_function):
    if activation_function == 1:
        return np.maximum(0, Z)
    if activation_function == 2:
        return np.maximum(0.01 * Z, Z)
    if activation_function == 3:
        return np.where(Z > 0, Z, 1 * (np.exp(Z) - 1))
    if activation_function == 4:
        lam=1.0507; alpha=1.6732
        return np.where(Z > 0, Z, alpha*(np.exp(Z)-1)) * lam
    if activation_function == 5:
        return 1 / (1 + np.exp(-Z))
    if activation_function == -1: # Special case: use for f(x) = x
        return Z


# Derivatives of the activation funciton
def deriva_act(Z, activation_function):
    if activation_function == 1:
        return np.where(Z >= 0, 1, 0)
    if activation_function == 2:
        return np.where(Z >= 0, 1, 0.01)
    if activation_function == 3:
        return np.where(Z >= 0, 1, 0.01 * np.exp(Z))
    if activation_function == 4:
        lam=1.0507; alpha=1.6732
        return np.where(Z > 0, 1, alpha*np.exp(Z)) * lam
    if activation_function == 5:
        s = act_function(Z, activation_function)
        return s * (1 - s)        


# Initialize one-wieght    
def iniW(next,prev):
    r = np.sqrt(6/(next + prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
    
    
# STEP 1: Feed-forward of DAE
def dae_forward(dae, X, params):
    layers = dae['layers']
    dae['A'][0] = X

    for l in range(1, layers):
        dae['Z'][l] = (dae['W'][l-1] @ dae['A'][l-1])
        dae['A'][l] = act_function(dae['Z'][l], params['dae_enc_act_func'])

    return dae['A'][-1]
        

# STEP 2: Feed-Backward for DAE
def dae_gradW(dae, params):
    act_func = params['dae_enc_act_func']

    layers = dae['layers']
    gradW = [None] * len(dae['W'])
    for l in reversed(range(1, layers)): # example l: 1 .. 6
        if l == (layers - 1): # output layer
            e = (dae['A'][-1] - dae['A'][0])
            delta = e * deriva_act(dae['Z'][l], act_func)
            gradW[l-1] = delta @ dae['A'][l-1].T
        else: # hidden layers
            delta = (dae['W'][l].T @ delta) * deriva_act(dae['Z'][l], act_func)

            if l == 1: # first layer
                gradW[l-1] = delta @ dae['A'][0].T
            else:
                gradW[l-1] = delta @ dae['A'][l-1].T
    
    return gradW


# Update DAE's weight via mAdam
def dae_updW_madam(dae, gradW, params):
    learning_rate = params['dae_learning_rate']
    beta_1, beta_2, epsilon = 0.9, 0.999, 10e-8
   
    for l in range(len(gradW)):
        dae['V'][l] = (beta_1 * dae['V'][l]) + ((1 - beta_1) * gradW[l])
        dae['S'][l] = (beta_2 * dae['S'][l]) + ((1 - beta_2) * np.square(gradW[l]))
        gAdam = (np.sqrt(1 - beta_2) / (1 - beta_1)) * ((dae['V'][l]) / (np.sqrt(dae['S'][l] + epsilon)))
        dae['W'][l] = dae['W'][l] - (learning_rate * gAdam)
    
    return dae['W']

# Softmax forward
def sft_forward(ann, X, params):
    layers = ann['layers']
    
    ann['A'][0] = X
    for l in range(1, layers):
        ann['Z'][l] = (ann['W'][l-1] @ ann['A'][l-1])
        if l != layers: # hidden layers
            ann['A'][l] = act_function(ann['Z'][l], params['dae_enc_act_func'])
        else: # output layer
            ann['A'][l] = softmax(ann['Z'][l])
    return ann['A'][-1]


# Softmax's gradient
def gradW_softmax(ann, Y, params):
    minibatch_size = params['sft_minibatch_size']    
    gradW = -(1/minibatch_size) * ((Y - ann['A'][-1]) @ ann['A'][0].T)
    return gradW


# Update Softmax's weight via mAdam
def updW_sft_mAdam(ann, gradW, params):
    learning_rate = params['sft_learning_rate']
    beta_1, beta_2, epsilon = 0.9, 0.999, 10e-8
   
    ann['V'] = (beta_1 * ann['V']) + ((1 - beta_1) * gradW)
    ann['S'] = (beta_2 * ann['S']) + ((1 - beta_2) * np.square(gradW))
    gAdam = (np.sqrt(1 - beta_2) / (1 - beta_1)) * ((ann['V']) / (np.sqrt(ann['S'] + epsilon)))
    ann['W'][-1] = ann['W'][-1] - (learning_rate * gAdam)
    
    return ann['W'][-1]


# Softmax cost calculation
def calculate_sft_cost(Y, Y_pred, params):
    minibatch_size = params['sft_minibatch_size']
    log_Y_pred = np.log(Y_pred)
    log_Y_pred[log_Y_pred == -np.inf] = 0
    cost = -np.sum(np.sum(Y * log_Y_pred, axis=0) / Y.shape[0]) / minibatch_size
    return cost


# Calculate Softmax
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return(exp_z/exp_z.sum(axis=0, keepdims=True))


# save weights SAE and cost of Softmax
def save_w_dl(W, costs):
    np.savez("W_snn.npz", *W)
    np.savetxt("costo.csv", costs, fmt="%.10f")


# TODO: delete before sending homework
def plot(X, path_to_save, labels=None, title=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for i, x in enumerate(X):
        ax.plot(range(0, len(x)), x, linewidth=1)

    plt.savefig(path_to_save)
    plt.show()
    

# TODO: delete before sending homework
def plot_this(X, path_to_save, labels=None, title=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for i, x in enumerate(X):
        ax.plot(range(0, len(x)), x,
                linewidth=1,
                label=labels[i] if labels[i] else f'{i+1}')

    # Enable legend
    ax.legend()
    ax.set_title(title) if title else ''
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')

    plt.savefig(path_to_save)
    plt.show()