import numpy as np
import numpy.random
import math
from matplotlib import pyplot

N=3000;
M11 = [0, 5];
M12 = [5, 0];
X1 = np.zeros((2,N));
for i in range(0,N):
	tmp = np.random.rand();
	if tmp>0.5:
		X1[0,i] = numpy.random.normal(scale=0.5)+M11[0];
		X1[1,i] = numpy.random.normal(scale=2)+M11[1];
	else:
		X1[0,i] = numpy.random.normal(scale=2)+M12[0];
		X1[1,i] = numpy.random.normal(scale=0.5)+M12[1];
pyplot.scatter(X1[0],X1[1]);

M2 = [6.5 ,6.5];
X2 = np.zeros((2,N));
X2[0,:] = numpy.random.normal(scale=1,size=(N))+M2[0];
X2[1,:] = numpy.random.normal(scale=1,size=(N))+M2[1];
pyplot.scatter(X2[0],X2[1]);
#pyplot.show();


Z0 = np.concatenate([np.ones((N)),-np.ones((N))]);
Z1 = np.concatenate([X1,-X2],axis=1);
Z2 = np.concatenate([X1*X1,-X2*X2],axis=1);
Z3 = np.concatenate([2*X1[0,:]*X1[1,:],-2*X2[0,:]*X2[1,:]]);
Z = np.zeros((6,2*N));
Z[0,:] = Z0;
Z[1:3,:] = Z1;
Z[3:5,:] = Z2;
Z[5,:] = Z3;
Gamma = np.concatenate([np.ones((N)),1*np.ones((N))]);
W = ((np.linalg.inv(Z.dot(np.transpose(Z)))).dot(Z)).dot(Gamma);

v0 = W[0];
V = W[1:3];
Q = np.zeros((2,2));
Q[0,0] = W[3];
Q[1,1] = W[4];
Q[0,1] = W[5];
Q[1,0] = W[5];
x = np.linspace(-4,11,100);
y = np.linspace(-4,11,100);
mes = np.zeros((2,100,100));
[mes[0,:,:], mes[1,:,:]] = np.meshgrid(x,y);
z = np.zeros((100,100));
for i in range(0,100):
	for ii in range(0,100):
		z[i,ii] = ((np.transpose(mes[:,i,ii])).dot(Q)).dot(mes[:,i,ii])+V.reshape((1,2)).dot( mes[:,i,ii])+v0;
#fig = pyplot.figure(2);
#ax = fig.add_subplot(111, projection = '3d')
#ax.plot_surface(mes[0,:,:], mes[1,:,:], z,cmap='coolwarm')

pyplot.contour(mes[0,:,:], mes[1,:,:], z,[0]);
pyplot.show();