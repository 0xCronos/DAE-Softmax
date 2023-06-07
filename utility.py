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
    for l in range(len(cnf_dae[8:])):
        params[f'encoder{l+1}_nodes'] = int(cnf_dae[l])
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


def normalize_data(X):
    for i in range(len(X)):
        X[i] = normalize_var(X[i])
    return X

def normalize_var(var, a=0.01, b=0.99):
    x_min = np.min(var)
    x_max = np.max(var)
    for i in range(len(var)):
        var[i] = (var[i] - x_min) / (x_max - x_min) * (b - a)  + a
    return var
        

def plot(X, path_to_save, labels=None, title=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for i, x in enumerate(X):
        ax.plot(range(0, len(x)), x, linewidth=1)

    plt.savefig(path_to_save)
    plt.show()


# # Initialize one-wieght    
# def iniW(next,prev):
#     r = np.sqrt(6/(next+ prev))
#     w = np.random.rand(next,prev)
#     w = w*2*r-r    
#     return(w)
    
# # STEP 1: Feed-forward of AE
# def dae_forward(x,...):
#     ...
#     return()    


# #Activation function
# def act_sigmoid(z):
#     return(1/(1+np.exp(-z)))   
# # Derivate of the activation funciton
# def deriva_sigmoid(a):
#     return(a*(1-a))
# # STEP 2: Feed-Backward for DAE
# def gradW(a,w2):   
#     ...
#     ...    
#     return(...)        

# # Update DAE's weight via mAdam
# def updW_madam():
#         ...    
#     return(...v)
# # Update Softmax's weight via mAdam
# def updW_sft_rmsprop(w,v,gw,mu):
#     ...    
#     return(w,v)

# # Softmax's gradient
# def gradW_softmax(x,y,a):        
#     ya   = y*np.log(a)
#     ...    
#     return(gW,Cost)

# # Calculate Softmax
# def softmax(z):
#         exp_z = np.exp(z-np.max(z))
#         return(exp_z/exp_z.sum(axis=0,keepdims=True))


# # save weights DL and costo of Softmax
# def save_w_dl(...):    
#     ...
    