import numpy as np
import utility as ut


def save_measure(cm, fsc):
    np.savetxt("cmatriz.csv", cm, fmt="%d")
    np.savetxt("fscores.csv", fsc, fmt="%.10f")


#load data for testing
def load_data_tst():
    X = np.loadtxt('X_test.csv', delimiter=',')
    Y = np.loadtxt('Y_test.csv', delimiter=',')
    return X.T, Y.T
    

#load weight of the DL in numpy format
def load_w_dl():
    data = np.load("w_snn.npz")
    return [data[key] for key in data.keys()]

def create_ann(W, X):
    A = [None] * (len(W) + 1)
    Z = [None] * (len(W) + 1)
    A[0] = X
    return {'W': W, 'A': A, 'Z': Z, 'layers': len(W)+1}


def metricas(Y, Y_predicted):
    cm = confusion_matrix(Y, Y_predicted)
    precision = cm.diagonal() / cm.sum(axis=0)
    recall = cm.diagonal() / cm.sum(axis=1)
    fsc = np.nan_to_num(2 * ((precision * recall) / (precision + recall)), nan=0)    
    fsc = np.append(fsc, np.mean(fsc))
    return cm, fsc
    
    
# Confusion matrix
def confusion_matrix(Y, Y_predicted):
    n_class = Y.shape[0]
    cm = np.zeros((n_class, n_class), dtype=int)
    for j in range(Y.shape[1]):
        max_index_real = np.argmax(Y[:, j])
        max_index_pred = np.argmax(Y_predicted[:, j])
        cm[max_index_real, max_index_pred] += 1
    return cm


# Beginning ...
def main():
    params = ut.load_config()
    Xt, Yt  = load_data_tst()
    W = load_w_dl()
    ann = create_ann(W, Xt)
    Y_pred = ut.sft_forward(ann, Xt, params)
    cm, fsc = metricas(Yt, Y_pred)
    save_measure(cm, fsc)
	

if __name__ == '__main__':   
	 main()
