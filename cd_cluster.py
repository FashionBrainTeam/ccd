#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:45:03 2018

@author: inesarous
"""

import numpy as np
import sys
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from numpy import linalg as LA
#from sklearn.cluster import KMeans
from numpy.linalg import matrix_rank
from scipy import stats

def SSV(x,n,m):
	pos = -1
	z = np.ones((n,1))
	v = (np.dot(x,np.transpose(np.sum(x, axis=0)))- np.sum(x*x,axis=1)).reshape((n,1))
	#print v
	var_bool=True
	while var_bool or (pos!=-1):
		var_bool=False
		#Change sign
		if pos!=-1:
			z[pos] = z[pos]*-1		
		#Determine v
		v=v+(2*z[pos]*(np.dot(x,x[pos]))).reshape(n,1)
		v[pos]=v[pos]-(2*z[pos]*(np.dot(x[pos],x[pos])))            
		#Search next element
		val=z*v
		if val[val<0].size!=0:
			pos=np.argmin(val)
		else:
			pos=-1
	return z
	
#Calculate centroid decomposition 
def CD(x,n,m,k):
	L =np.zeros((n,k))
	R =np.zeros((m,k))
	#print R[:,0]
	for i in range (0,k):
		z = SSV(x,n,m)    
		R[:,[i]]  = np.dot(np.transpose(x),z)/LA.norm(np.dot(np.transpose(x),z))
		L[:,i] = np.dot(x,R[:,i])
		x = x - np.dot(L[:,[i]],np.transpose(R[:,[i]]))
	return L,R


def CCD(x,n,m,k):
	Lk,Rk = CD(x,n,m,(m-k))
	print Rk
	E=(Lk>=0)
	classes=np.zeros((n, 1))
	for i in range (0,(m-k)):
	    classes=classes+(2**(i))*(E[:,i].reshape(n,1))  
	return classes

input_fashion=pd.read_csv('/Volumes/GoogleDrive/My Drive/Phd/data/Fashion/Zalando_data_M3/modified_data/timeseries_nitems/region1_fs.csv')
best_combination=pd.concat([input_fashion.iloc[:,1],input_fashion.iloc[:,4],input_fashion.iloc[:,6],input_fashion.iloc[:,8]],axis=1)
input_matrix=stats.zscore(best_combination.values)
#b, a = signal.butter(4, 0.001)
#zi = signal.lfilter_zi(b, a)
#z= signal.lfilter(b, a, input_matrix)

#input_matrix=savgol_filter(input_matrix1, window_length=3, polyorder=2, mode='nearest')
classes=CCD(input_matrix,input_matrix.shape[0],input_matrix.shape[1],3)
df=pd.concat([pd.DataFrame(data=input_matrix),pd.DataFrame(data=classes,columns=['ccd'])], axis=1)

fig=plt.figure(figsize=(20,10))
plt.plot(df,linestyle='solid',marker='None')
plt.ylabel('# items', fontsize=20)
plt.xlabel('timestamps', fontsize=20)
fig.suptitle('Fashion time series', fontsize=20)
ax1 = df.cumsum().plot()
lines, labels = ax1.get_legend_handles_labels()
fig.legend(lines[:5], labels[:5], loc='best')  # legend for first two lines only
plt.ion()
plt.show()
## Define main method that calls other functions
#def main():
#	inputfile=sys.argv[1]       #'./Input/example.txt'
# 	outputfile=sys.argv[2]     #'./Result/classes.txt'  
# 	n =int(sys.argv[3])                #input_matrix.shape[0]
#	m =int(sys.argv[4])                 #input_matrix.shape[1]
#	k=int(sys.argv[5])                  #k=2    
#	input_matrix=np.loadtxt(inputfile,delimiter=' ')
#	#input_matrix=stats.zscore(input_matrix)
#	#classify the matrix using CD
#	classes=CCD(input_matrix,n,m,k)
#	np.savetxt(outputfile,classes,fmt='%.2f')
#    
#
## Execute main() function
#if __name__ == '__main__':
#    main()
