import numpy as np
import numpy.random
import math
from matplotlib import pyplot

N=3000;
M11 = [0, 3];
M12 = [3, 0];
X1 = np.zeros((2,N));
for i in range(0,N):
	tmp = np.random.rand();
	if tmp>0.5:
		X1[0,i] = numpy.random.normal(scale=0.5)+M11[0];
		X1[1,i] = numpy.random.normal(scale=1.5)+M11[1];
	else:
		X1[0,i] = numpy.random.normal(scale=1.5)+M12[0];
		X1[1,i] = numpy.random.normal(scale=0.5)+M12[1];
#pyplot.scatter(X1[0],X1[1]);

M2 = [4.7 ,4.7];
X2 = np.zeros((2,N));
X2[0,:] = numpy.random.normal(scale=0.7,size=(N))+M2[0];
X2[1,:] = numpy.random.normal(scale=0.7,size=(N))+M2[1];
#pyplot.scatter(X2[0],X2[1]);
#pyplot.show();

X = np.concatenate([X1,X2],axis=1);
clas = numpy.random.randint(0,2,2*N);
#clas = np.concatenate([np.ones((int(0.1*N)))-1,numpy.random.randint(0,4,int(0.9*N)),np.ones((int(0.1*N))),numpy.random.randint(0,4,int(0.9*N))]);

mean = np.zeros((2,2));
cov = np.zeros((2,2,2));
for i in range(0,2):
	mean[:,i] = X[:,clas==i].mean(axis=1);
	cov[i,:,:] = np.cov(X[:,clas==i]);

lmax = 10;
izm = 3;
l=0;
while (l<lmax) and izm>2:
	l += 1;
	izm = 0;
	for i in range(0,2*N):
		d = np.zeros((2));
		for  ii in range(0,2):
			#d[ii] = sum((X[:,i]-mean[:,ii])**2);
			d[ii] = math.log(np.linalg.det(cov[ii,:,:]))/2+(np.transpose((X[:,i]-mean[:,ii])).dot(np.linalg.inv(cov[ii,:,:]))).dot(X[:,i]-mean[:,ii]);
		cl = np.argmin(d);
		if (cl != clas[i]):# and (sum(clas==clas[i])>1):
			izm += 1;
			clas[i] = cl;
	for i in range(0,2):
		mean[:,i] = X[:,clas==i].mean(axis=1);
		cov[i,:,:] = np.cov(X[:,clas==i]);

print(l);
for i in range(0,2):
	pyplot.scatter(X[0,clas==i],X[1,clas==i]);
pyplot.show();
a=1;