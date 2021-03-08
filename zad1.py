import numpy as np
from numpy.random import *
from skimage import *
from skimage.io import *
from scipy.stats import *
from matplotlib import pyplot

def extract_features(i,letter,bu):
	im = img_as_float(imread("Slova/baza"+letter+str(i+1).zfill(3)+".bmp"));
	im_bin = np.zeros_like(im);
	im_bin[im>0.92]=1;
	shape = im_bin.shape;
	poc1 = poc2 = 10;
	kraj1 = shape[1]-10;
	kraj2 = shape[0]-10;
	while sum(im_bin[:,poc1])<0.91*shape[0]:
		poc1 +=1;
	while sum(im_bin[poc2,:])<0.91*shape[1]:
		poc2 +=1;
	while sum(im_bin[:,kraj1])<0.91*shape[0]:
		kraj1 -=1;
	while sum(im_bin[kraj2,:])<0.91*shape[1]:
		kraj2 -=1;

	im_crop1 = im_bin[poc2:kraj2+1,poc1:kraj1+1];
	shape = im_crop1.shape;
	poc1 = poc2 = 0;
	kraj1 = shape[1]-1;
	kraj2 = shape[0]-1;
	while sum(im_crop1[:,poc1])>0.96*shape[0] and poc1<shape[1]-1:
		poc1 +=1;
	while sum(im_crop1[poc2,:])>0.96*shape[1] and poc1<shape[0]-1:
		poc2 +=1;
	while sum(im_crop1[:,kraj1])>0.96*shape[0] and kraj1>poc1:
		kraj1 -=1;
	while sum(im_crop1[kraj2,:])>0.96*shape[1] and kraj2>poc2:
		kraj2 -=1;
	#print(poc1,poc2,kraj1,kraj2);
	im_crop = im_crop1[poc2:kraj2+1,poc1:kraj1+1];

	shape1 = im_crop.shape;

	feat1 = sum(im_crop[:,0:int(shape1[1]/2)].flatten())/sum(im_crop.flatten());
	#feat1 = sum(im_crop[0:int(shape1[0]/2),:].flatten())/sum(im_crop.flatten());
	#feat1 = 0;
	#for i in range(0,shape1[0]):
	#	feat1 += abs(sum(im_crop[i,0:int(shape1[1]/2)])-sum(im_crop[i,int(shape1[1]/2):shape1[1]]));
	#feat1 /= sum(im_crop.flatten());
	
	feat2 = 0;
	for i in range(0,shape1[1]):
		feat2 += abs(sum(im_crop[0:int(shape1[0]/2),i])-sum(im_crop[int(shape1[0]/2):shape1[0],i]));
	feat2 /= sum(im_crop.flatten());
	feat3 = sum(im_crop[int(shape1[0]/4):int(3*shape1[0]/4),int(shape1[1]/4):int(3*shape1[1]/4)].flatten())/sum(im_crop.flatten());
	if(bu):
		imshow(im);
		show();

	return [feat1,feat2,feat3];

nf = 3; #broj featurea
features = np.zeros((5,nf,100));
letters = ['A','E','I','O','U'];
for i in range(0,100):
	for ii in range (0,5):
		features[ii,:,i] = extract_features(i,letters[ii],False);

cov1 = np.zeros((5,nf,nf));
mean1 = np.zeros((5,nf));
for i in range(0,5):
	cov1[i,:,:] = np.cov(features[i,:,:]);
	mean1[i,:] = np.mean(features[i,:,:],axis=1);

conf_matrix = np.zeros((5,5));
f = np.zeros((5));
for i in range(100,120):
	for ii in range(0,5):
		feature = extract_features(i,letters[ii],False);
		for iii in range(0,5):
			f[iii] = multivariate_normal.pdf(feature,mean = mean1[iii,:],cov = cov1[iii,:,:]);
		ind = np.where(f==max(f))[0];
		#print(ind,ii);
		#if ind !=ii:
		#	extract_features(i,letters[ii],True);
		conf_matrix[ind,ii] += 1;

print(conf_matrix);
a=1;