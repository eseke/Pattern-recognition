import numpy as np
import numpy.random
import math
from matplotlib import pyplot
from scipy.stats.stats import Ttest_1sampResult

N=300;
M1 = [2 ,2];
X1 = np.zeros((2,N));
X1[0,:] = numpy.random.normal(scale=0.65,size=(N))+M1[0];
X1[1,:] = numpy.random.normal(scale=0.65,size=(N))+M1[1];
#pyplot.scatter(X1[0],X1[1]);

M2 = [8 ,2];
X2 = np.zeros((2,N));
X2[0,:] = numpy.random.normal(scale=0.65,size=(N))+M2[0];
X2[1,:] = numpy.random.normal(scale=0.65,size=(N))+M2[1];
#pyplot.scatter(X2[0],X2[1]);

M3 = [2 ,8];
X3 = np.zeros((2,N));
X3[0,:] = numpy.random.normal(scale=0.65,size=(N))+M3[0];
X3[1,:] = numpy.random.normal(scale=0.65,size=(N))+M3[1];
#pyplot.scatter(X3[0],X3[1]);

M4 = [8 ,8];
X4 = np.zeros((2,N));
X4[0,:] = numpy.random.normal(scale=0.65,size=(N))+M4[0];
X4[1,:] = numpy.random.normal(scale=0.65,size=(N))+M4[1];
#pyplot.scatter(X4[0],X4[1]);
#pyplot.show();


X = np.concatenate([X1,X2,X3,X4],axis=1);
#pyplot.scatter(X[0],X[1]);
#pyplot.show();

clas = numpy.random.randint(0,4,4*N); 
#clas = np.concatenate([np.ones((int(0.5*N)))-1,numpy.random.randint(0,4,int(0.5*N)),np.ones((int(0.5*N))),numpy.random.randint(0,4,int(0.5*N)),np.ones((int(0.5*N)))*2,numpy.random.randint(0,4,int(0.5*N)),np.ones((int(0.5*N)))*3,numpy.random.randint(0,4,int(0.5*N))]);
#for i in range(0,4):
#	pyplot.scatter(X[0,clas==i],X[1,clas==i]);
#pyplot.show();
mean = np.zeros((2,4));
cov = np.zeros((4,2,2));
for i in range(0,4):
	mean[:,i] = X[:,clas==i].mean(axis=1);
	cov[i,:,:] = np.cov(X[:,clas==i]);

lmax = 100;
izm = 3;
l=0;
while (l<lmax) and izm>2:
	l += 1;
	izm = 0;
	for i in range(0,4*N):
		d = np.zeros((4));
		for  ii in range(0,4):
			d[ii] = sum((X[:,i]-mean[:,ii])**2);
		cl = np.argmin(d);
		if (cl != clas[i]):# and (sum(clas==clas[i])>1):
			izm += 1;
			clas[i] = cl;
	for i in range(0,4):
		mean[:,i] = X[:,clas==i].mean(axis=1);
		cov[i,:,:] = np.cov(X[:,clas==i]);

print(l);
for i in range(0,4):
	pyplot.scatter(X[0,clas==i],X[1,clas==i]);
pyplot.show();
a=1;
