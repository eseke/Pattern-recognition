import numpy as np
import numpy.random
import math
from matplotlib import pyplot

N=3000;
M1 = [2 ,2];
X1 = np.zeros((2,N));
X1[0,:] = numpy.random.normal(scale=1.1,size=(N))+M1[0];
X1[1,:] = numpy.random.normal(scale=0.6,size=(N))+M1[1];
pyplot.scatter(X1[0],X1[1]);

M2 = [4.5 ,4.5];
X2 = np.zeros((2,N));
X2[0,:] = numpy.random.normal(scale=0.6,size=(N))+M2[0];
X2[1,:] = numpy.random.normal(scale=1.1,size=(N))+M2[1];
pyplot.scatter(X2[0],X2[1]);
#pyplot.show();

cov1_est = np.cov(X1[:,:2100]);
M1_est = np.mean(X1[:,:2100],axis=1);
cov2_est = np.cov(X2[:,:2100]);
M2_est = np.mean(X2[:,:2100],axis=1);

s = np.linspace(0,1,31);
y1 = np.zeros((900));
y2 = np.zeros((900));
Nw_s =np.zeros((31));
v0_s = np.zeros((31));
for i in range(0,31):
	V = (s[i]*cov1_est+(1-s[i])*cov2_est).dot(M2_est-M1_est);
	for ii in range(0,900):
		y1[ii] = V.reshape((1,2)).dot(X1[:,ii+2100]);
		y2[ii] = V.reshape((1,2)).dot(X2[:,ii+2100]);
	v0_min = -max(max(y1),max(y2));
	v0_max = -min(min(y1),min(y2));
	v0 = np.linspace(v0_min,v0_max,100);
	Nw = np.zeros((100));
	for ii in range(0,100):
		Nw_curr = 0;
		for iii in range(0,900):
			if y1[iii]> -v0[ii]:
				Nw_curr += 1;
			if y2[iii]< -v0[ii]:
				Nw_curr += 1;
		Nw[ii] = Nw_curr;
		a=1;
	#pyplot.plot(v0,Nw);
	#pyplot.show();
	v0_s[i] = v0[np.argmin(Nw)];
	Nw_s[i] = Nw[np.argmin(Nw)];
	a = 1;



v0 = v0_s[np.argmin(Nw_s)];
s_min = s[np.argmin(Nw_s)];
V = (s_min*cov1_est+(1-s_min)*cov2_est).dot(M2_est-M1_est);
x = np.linspace(-2,7,100);
y = np.linspace(0,9,100);
mes = np.zeros((2,100,100));
[mes[0,:,:], mes[1,:,:]] = np.meshgrid(x,y);
z = np.zeros((100,100));
for i in range(0,100):
	for ii in range(0,100):
		z[i,ii] = V.reshape((1,2)).dot( mes[:,i,ii])+v0;
#fig = pyplot.figure(2);
#ax = fig.add_subplot(111, projection = '3d')
#ax.plot_surface(mes[0,:,:], mes[1,:,:], z,cmap='coolwarm')

pyplot.contour(mes[0,:,:], mes[1,:,:], z,[0]);

pyplot.figure(2);
pyplot.plot(s,Nw_s);
pyplot.xlabel('s');
pyplot.ylabel('Nw_min');

pyplot.figure(3);
M1 = [1 ,1];
X1 = np.zeros((2,N));
X1[0,:] = numpy.random.normal(scale=1.5,size=(N))+M1[0];
X1[1,:] = numpy.random.normal(scale=1.5,size=(N))+M1[1];
pyplot.figure(1);
pyplot.scatter(X1[0],X1[1]);

M2 = [7 ,7];
X2 = np.zeros((2,N));
X2[0,:] = numpy.random.normal(scale=1,size=(N))+M2[0];
X2[1,:] = numpy.random.normal(scale=1,size=(N))+M2[1];
pyplot.scatter(X2[0],X2[1]);
#pyplot.show();

Z0 = np.concatenate([np.ones((N)),-np.ones((N))]);
Z1 = np.concatenate([X1,-X2],axis=1);
Z = np.zeros((3,2*N));
Z[0,:] = Z0;
Z[1:3,:] = Z1;
Gamma1 = np.ones((N));
#Gamma1[(X1[0,:]>2)*(X1[1,:]>2)] = 10;
Gamma2 = np.ones((N));
Gamma = np.concatenate([Gamma1,Gamma2]);
W = ((np.linalg.inv(Z.dot(np.transpose(Z)))).dot(Z)).dot(Gamma);

v0 = W[0];
V = W[1:3];
x = np.linspace(-4,11,100);
y = np.linspace(-4,11,100);
mes = np.zeros((2,100,100));
[mes[0,:,:], mes[1,:,:]] = np.meshgrid(x,y);
z = np.zeros((100,100));
for i in range(0,100):
	for ii in range(0,100):
		z[i,ii] = V.reshape((1,2)).dot( mes[:,i,ii])+v0;
#fig = pyplot.figure(2);
#ax = fig.add_subplot(111, projection = '3d')
#ax.plot_surface(mes[0,:,:], mes[1,:,:], z,cmap='coolwarm')

pyplot.contour(mes[0,:,:], mes[1,:,:], z,[0]);
#pyplot.show();

pyplot.figure(4);
pyplot.scatter(X1[0],X1[1]);
pyplot.scatter(X2[0],X2[1]);

Z0 = np.concatenate([np.ones((N)),-np.ones((N))]);
Z1 = np.concatenate([X1,-X2],axis=1);
Z = np.zeros((3,2*N));
Z[0,:] = Z0;
Z[1:3,:] = Z1;
Gamma1 = np.ones((N));
Gamma1[(X1[0,:]>2)*(X1[1,:]>2)] = 8;
Gamma2 = np.ones((N));
Gamma = np.concatenate([Gamma1,Gamma2]);
W = ((np.linalg.inv(Z.dot(np.transpose(Z)))).dot(Z)).dot(Gamma);

v0 = W[0];
V = W[1:3];
x = np.linspace(-4,11,100);
y = np.linspace(-4,11,100);
mes = np.zeros((2,100,100));
[mes[0,:,:], mes[1,:,:]] = np.meshgrid(x,y);
z = np.zeros((100,100));
for i in range(0,100):
	for ii in range(0,100):
		z[i,ii] = V.reshape((1,2)).dot( mes[:,i,ii])+v0;
#fig = pyplot.figure(2);
#ax = fig.add_subplot(111, projection = '3d')
#ax.plot_surface(mes[0,:,:], mes[1,:,:], z,cmap='coolwarm')

pyplot.contour(mes[0,:,:], mes[1,:,:], z,[0]);
pyplot.show();
