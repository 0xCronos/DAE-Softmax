import pandas as pd
import numpy as np
import utility as ut

#load data for testing
def load_data_tst():
    ...    
    return(xv,yv)    


#load weight of the DL in numpy format
def load_w_dl():
    ...    
    return(W)    



# Feed-forward of the DL
def forward_dl(x,W):        
    ...    
    return(zv)


# Métrica
def metricas(x,y):
    cm     = confusion_matrix(x,y)
    ...   
    return(cm,Fscore)
    
#Confusuon matrix
def confusion_matrix(y,z):
    ...    
    return(cm)



# Beginning ...
def main():		
	xv,yv  = load_data_tst()
	W      = load_w_dl()
	zv     = forward_dl(xv,W)      		
	cm,Fsc = metricas(yv,zv) 		
	print(Fsc*100)
	print('Fsc-mean {:.5f}'.format(Fsc.mean()*100))
	

if __name__ == '__main__':   
	 main()
