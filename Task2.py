import numpy as np
from matplotlib import pyplot
from scipy.stats import multivariate_normal
from math import log

odb = np.zeros((2,2,500));

M11 = [1, 1];
M12 = [6, 4];
M21 = [7, -4];
M22 = [6, 0];
S11 = [[3,1.1],[1.1,2]];
S12 = [[3,-0.8],[-0.8,1.5]];
S21 = [[2,1.1],[1.1,4]];
S22 = [[3,0.8],[0.8,0.5]];

[L11t,F11] = np.linalg.eig(S11);
[L12t,F12] = np.linalg.eig(S12);
[L21t,F21] = np.linalg.eig(S21);
[L22t,F22] = np.linalg.eig(S22);

L11 = np.eye(2)*L11t;
L12 = np.eye(2)*L12t;
L21 = np.eye(2)*L21t;
L22 = np.eye(2)*L22t;


for i in range(0,500):
	tmp = np.random.uniform();
	if tmp<0.6:
		odb[0,:,i] = F11.dot((L11)**(1/2)).dot(np.random.normal(size=(2)))+M11;
	else:
		odb[0,:,i] = F12.dot((L12)**(1/2)).dot(np.random.normal(size=(2)))+M12;
	if tmp<0.55:
		odb[1,:,i] = F21.dot((L21)**(1/2)).dot(np.random.normal(size=(2)))+M21;
	else:
		odb[1,:,i] = F22.dot((L22)**(1/2)).dot(np.random.normal(size=(2)))+M22;

pyplot.figure(1);
pyplot.scatter(odb[0,0,:],odb[0,1,:]);
pyplot.scatter(odb[1,0,:],odb[1,1,:]);


f1 = np.zeros((171,161));
f2 = np.zeros((171,161));
h = np.zeros((171,161));
x2 = np.linspace(-4,12,161);
x1 = np.linspace(-10,7,171);
for ii in range(0,161):
	for i in range(0,171):
		f11 = multivariate_normal.pdf([x2[ii],x1[i]],M11,S11)*0.01;
		f12 = multivariate_normal.pdf([x2[ii],x1[i]],M12,S12)*0.01;
		f21 = multivariate_normal.pdf([x2[ii],x1[i]],M21,S21)*0.01;
		f22 = multivariate_normal.pdf([x2[ii],x1[i]],M22,S22)*0.01;
		f1[i,ii] = 0.6*f11+0.4*f12;
		f2[i,ii] = 0.55*f21+0.45*f22;
		h[i,ii] = -log(f1[i,ii])+log(f2[i,ii]);
f1 /=sum(f1.flatten());
f2 /=sum(f2.flatten());

X, Y = np.meshgrid( x2, x1)
#pyplot.contour(X,Y,f1,[max(f1.flatten())*0.3,max(f1.flatten())*0.6,max(f1.flatten())*0.8]);
#pyplot.contour(X,Y,f2,[max(f2.flatten())*0.05,max(f2.flatten())*0.2,max(f2.flatten())*0.8]);
pyplot.contour(X,Y,h,[0]);
#fig = pyplot.figure(2);
#ax = fig.add_subplot(111, projection = '3d')
#ax.plot_surface(X, Y, f1+f2,cmap='coolwarm')

g1= 0;
g2=0
for i in range(0,171):
	for ii in range(0,161):
		if h[i,ii]>0:
			g1 += 0.5*f1[i,ii];
		else:
			g2 += 0.5*f2[i,ii];
print(g1*100,g2*100,(g1+g2)*100)

error = np.zeros((41));
mis = np.linspace(0.5,4.5,41);
for i in range(0,41):
	for ii in range(0,171):
		for iii in range(0,161):
			if h[ii,iii]>-log(mis[i]):
				error[i] += 0.5*f1[ii,iii];


wanted_e1 = 0.018;
mi = mis[abs(error-wanted_e1)==min(abs(error-wanted_e1))][0];
h += log(mi);
pyplot.figure(2);
pyplot.scatter(odb[0,0,:],odb[0,1,:]);
pyplot.scatter(odb[1,0,:],odb[1,1,:]);
pyplot.contour(X,Y,h,[0]);

#pyplot.figure(3);
#pyplot.plot(mis,error);
error_e2 = 0;
for ii in range(0,171):
	for iii in range(0,161):
		if h[ii,iii]<-log(3):
			error_e2 += 0.5*f2[ii,iii];
print('Vrednost parametra mi je '+str(mis[abs(error-wanted_e1)==min(abs(error-wanted_e1))][0])+' ukoliko zelimo daverovatnoca greska prvog tipa bude '+str(wanted_e1*100)+'%, dok je verovatnoca greske 2 tipa: '+str(error_e2*100)+'%')

h = np.zeros((171,161));
x1 = np.linspace(-10,7,171);
x2 = np.linspace(-4,12,161);
for i in range(0,171):
	for ii in range(0,161):
		f11 = multivariate_normal.pdf([x2[ii],x1[i]],M11,S11)*0.01;
		f12 = multivariate_normal.pdf([x2[ii],x1[i]],M12,S12)*0.01;
		f21 = multivariate_normal.pdf([x2[ii],x1[i]],M21,S21)*0.01;
		f22 = multivariate_normal.pdf([x2[ii],x1[i]],M22,S22)*0.01;
		f1 = 0.6*f11+0.4*f12;
		f2 = 0.55*f21+0.45*f22;
		h[i,ii] = -log(f1)+log(f2);

e1 = np.linspace(0.01,0.5,50);
e2 = np.linspace(0.01,0.5,50);
res1 = np.zeros((50,50));
for i in range(0,50):
	for ii in range(0,50):
		m=0;
		S=0;
		while S>-log((1-e1[i])/(e2[ii])):
			tmp = np.random.uniform();
			if tmp<0.6:
				odbirak = F11.dot((L11)**(1/2)).dot(np.random.normal(size=(2)))+M11;
			else:
				odbirak = F12.dot((L12)**(1/2)).dot(np.random.normal(size=(2)))+M12;
			f11 = multivariate_normal.pdf([odbirak[0],odbirak[1]],M11,S11)*0.01;
			f12 = multivariate_normal.pdf([odbirak[0],odbirak[1]],M12,S12)*0.01;
			f21 = multivariate_normal.pdf([odbirak[0],odbirak[1]],M21,S21)*0.01;
			f22 = multivariate_normal.pdf([odbirak[0],odbirak[1]],M22,S22)*0.01;
			f1 = 0.6*f11+0.4*f12;
			f2 = 0.55*f21+0.45*f22;
			h = -log(f1/f2);
			S += h*0.01;
			m += 1;
		res1[i,ii] = m;

fig = pyplot.figure(3);
X, Y = np.meshgrid( e1, e2)

ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, res1,cmap='coolwarm')
ax.set_xlabel('Greška prvog tipa');
ax.set_ylabel('Greška drugog tipa');
ax.set_zlabel('Broj odbiraka');

e1 = np.linspace(0.01,0.5,50);
e2 = np.linspace(0.01,0.5,50);
res1 = np.zeros((50,50));
for i in range(0,50):
	for ii in range(0,50):
		m=0;
		S=0;
		while S<-log((e1[i])/(1-e2[ii])):
			tmp = np.random.uniform();
			if tmp<0.55:
				odbirak = F21.dot((L21)**(1/2)).dot(np.random.normal(size=(2)))+M21;
			else:
				odbirak = F22.dot((L22)**(1/2)).dot(np.random.normal(size=(2)))+M22;
			f11 = multivariate_normal.pdf([odbirak[0],odbirak[1]],M11,S11)*0.01;
			f12 = multivariate_normal.pdf([odbirak[0],odbirak[1]],M12,S12)*0.01;
			f21 = multivariate_normal.pdf([odbirak[0],odbirak[1]],M21,S21)*0.01;
			f22 = multivariate_normal.pdf([odbirak[0],odbirak[1]],M22,S22)*0.01;
			f1 = 0.6*f11+0.4*f12;
			f2 = 0.55*f21+0.45*f22;
			h = -log(f1/f2);
			S += h*0.01;
			m += 1;
		res1[i,ii] = m;

fig = pyplot.figure(4);
X, Y = np.meshgrid( e1, e2)
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, res1,cmap='coolwarm')
ax.set_xlabel('Greška prvog tipa');
ax.set_ylabel('Greška drugog tipa');
ax.set_zlabel('Broj odbiraka');
pyplot.show()
