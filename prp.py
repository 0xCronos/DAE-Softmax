import numpy as np
import utility as ut


def get_frames(X, n_frames, frame_size):
    frames = []
    for i in range(n_frames):
        start_index = i * frame_size
        end_index = start_index + frame_size
        frames.append(list(X)[start_index:end_index])
    return frames


def get_amplitudes(frame):
    fft = np.abs(np.fft.fft(frame))
    return fft[:len(fft)//2]



def create_dtrn_dtst(X, Y, training_percentage):
    random_X, random_Y = ut.sort_data_random(X,Y)
    dtrn_amount = int(len(random_X) * training_percentage)
    
    
    dtrn = {
        'data': random_X[:dtrn_amount, :],
        'labels': random_Y[:dtrn_amount, :],
    }
    dtst = {
        'data': random_X[dtrn_amount:, :],
        'labels': random_Y[dtrn_amount:, :],
    }
    return dtrn, dtst


def create_binary_labels(features, params):
    n_classes = params['n_classes']
    n_frames = params['n_frames']
    n_vars = 4

    labels = np.zeros((len(features), n_classes), dtype=int)
    idx = np.arange(len(labels))
    
    labels[idx, idx // (n_vars * n_frames)] = 1

    return np.array(labels)


def create_features(data, params):
    n_frames = params['n_frames']
    frame_size = params['frame_size']
    
    features = []
    for _, X in enumerate(data):
        for col in range(X.shape[1]):
            frames = get_frames(X[:, col], n_frames, frame_size)
            amplitudes = list(map(get_amplitudes, frames))
            features.extend(amplitudes)
    features = ut.normalize_data(features)
    return np.array(features)


#Create new Data : Input and Label 
def create_input_label(data, params):
    features = create_features(data, params)
    labels = create_binary_labels(features, params)
    dtrn, dtst = create_dtrn_dtst(features, labels, params['trn_percentage'])
    return dtrn, dtst


#Save Data : training and testing
def save_data_csv(dtrn, dtst):
    np.savetxt("X_train.csv", dtrn['data'], delimiter=",")
    np.savetxt("Y_train.csv", dtrn['labels'], delimiter=",", fmt='%i')
    np.savetxt("X_test.csv", dtst['data'], delimiter=",")
    np.savetxt("Y_test.csv", dtst['labels'], delimiter=",", fmt='%i')


# Load data csv
def load_class_csv(params):
    n_classes = params['n_classes']
    data = [np.loadtxt(f'DATA/class{n+1}.csv', delimiter=',') for n in range(n_classes)]
    return data


# Beginning ...
def main():        
    params = ut.load_config()
    data = load_class_csv(params)
    dtrn, dtst = create_input_label(data, params)
    save_data_csv(dtrn, dtst)
    

if __name__ == '__main__':   
	main()

