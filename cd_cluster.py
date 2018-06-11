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


# Define main method that calls other functions
def main():
	inputfile=sys.argv[1]       #'./Input/example.txt'
 	outputfile=sys.argv[2]     #'./Result/classes.txt'  
 	n =int(sys.argv[3])                #input_matrix.shape[0]
	m =int(sys.argv[4])                 #input_matrix.shape[1]
	k=int(sys.argv[5])                  #k=2    
	input_matrix=np.loadtxt(inputfile,delimiter=' ')
	input_matrix=stats.zscore(input_matrix)
	#classify the matrix using CD
	classes=CCD(input_matrix,n,m,k)
	np.savetxt(outputfile,classes,fmt='%.2f')
    

# Execute main() function
if __name__ == '__main__':
    main()
