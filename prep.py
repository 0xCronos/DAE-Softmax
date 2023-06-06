import os
import pandas     as pd
import numpy      as np

# Crate new Data : Input and Label 
def create_input_label(data,param):
  labels =[]
  p_training = param[3]
  for i,value in enumerate(data):
      label = binary_label(i,value.shape[0],len(data))
      labels.append(label)
  Y = np.vstack(labels)
  X = np.vstack(data)

  M = np.concatenate((X, Y), axis=1)
  np.random.shuffle(M)

  split_index = int(M.shape[0] * p_training)

  trn_set, test_set = M[:split_index, :], M[split_index:, :]

  X_train, Y_train = trn_set[:, :X.shape[1]], trn_set[:, -Y.shape[1]:]
  X_test, Y_test = test_set[:, :X.shape[1]], test_set[:, -Y.shape[1]:]
  return X_train, Y_train, X_test, Y_test


# Save Data : training and testing
def save_data_cvs(xe,ye,xv,yv):
  dtrain = np.concatenate((xe,ye),axis=1)
  dtst= np.concatenate((xv,yv),axis=1)
  np.savetxt("train.csv", dtrain, delimiter=",", fmt="%f")
  np.savetxt("test.csv", dtst, delimiter=",", fmt="%f")
  return


# Binary Label
def binary_label(i, m, n):
    binary_array = np.zeros((m, n))
    binary_array[:, i] = 1
    return binary_array


# Load data csv
def load_class_csv():
  # Directorio donde se encuentran los archivos
    dir_path = "DATA/"

    # Obtener la lista de archivos en el directorio
    files = os.listdir(dir_path)

    # Filtrar los archivos que siguen el patrón "classX.csv"
    files = [f for f in files if f.startswith("class") and f.endswith(".csv")]

    # Leer cada archivo y almacenarlo en un dataframe
    dataframes = []
    for file in files:
        filepath = os.path.join(dir_path, file)
        df = np.loadtxt(filepath, delimiter=',')
        dataframes.append(df)

    return dataframes



# Configuration of the DAEs
def load_cnf_dae():      
    par = np.genfromtxt('cnf_dae.csv',delimiter=',')    
    return(par)



# Beginning ...
def main():        
    Param           = load_cnf_dae()	
    Data            = load_class_csv()
    Xe,Ye,Xv,Yv     = create_input_label(Data,Param)
    save_data_cvs(Xe,Ye,Xv,Yv)
    

if __name__ == '__main__':   
	 main()

