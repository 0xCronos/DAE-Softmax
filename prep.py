import pandas     as pd
import numpy      as np

# Crate new Data : Input and Label 
def create_input_label():
    ...  
  return(...)


# Save Data : training and testing
def save_data_csv(...):
    ...  
  return


# Binary Label
def binary_label():
  ...
  return(...)



# Load data csv
def load_class_csv():
    ...
  return(...) 


# Configuration of the DAEs
def load_cnf_dae():      
    par = np.genfromtxt('cnf_dae.csv',delimiter=',')    
    ...
    return(...)



# Beginning ...
def main():        
    Param           = load_cnf_dae()	
    Data            = load_class_csv(...)
    Xe,Ye,Xv,Yv     = create_input_label(...)
    save_data_cvs(Xe,Ye,Xv,Yv)
    

if __name__ == '__main__':   
	 main()

