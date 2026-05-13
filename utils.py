import numpy as np;
import os;
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def test_truth(data):
	counter=0
	x=0;
	y=0;
	while counter<100 and (x!=-1 or y!=-1):
		counter+=1
		x=int(input("x:"));
		y=int(input("y:"))
		dx= data[:,x]
		dy= data[:,y]
		plt.scatter(dx, dy)	
		plt.show()


def Standardize(data):
	dt_all = StandardScaler().fit(data).transform(data)
	return CleanMat(dt_all)


def F1_Or(gt,network):
	tp = np.sum((gt+network)==2)*1.0
	tn = np.sum((gt+network)==0)*1.0
	fn = 0
	fp = 0
	dims = gt.shape[1]
	if gt.shape!=network.shape:
		print("Matrices dont match, can't compute F1'")
		return 0
	for i in range(dims):
		for j in range(dims):
			if gt[i,j]==1 and network[i,j]==0:
				fn+=1.0
			if gt[i,j]==0 and network[i,j]==1:
				fp+=1.0

	F1=0
	if tp>0:
		precision = tp / (tp+fp)
		recall = tp / (tp+fn)
		F1 = 2 * (precision*recall) / (precision+recall)

	return np.round(F1,2)


def CleanMat(data,lim=3):
	#return data,target;
	mu_ = np.mean(data,axis=0);
	sd_ = np.std(data,axis=0);
	
	upper_limit= mu_ + lim*sd_;
	lower_limit= mu_ - lim*sd_;
	
	z1 = data[:,]<= upper_limit 
	z2 = data[:,]>=lower_limit;
	
	#print upper_limit;
	#print lower_limit;
	k1 =[];
	for r in range(len(z1)):
		k1.append( z1[r].all() and z2[r].all());
	
	return data[np.where(k1)];

def LoadData(filename,delim=","):
	print("File: ",filename)
	return np.genfromtxt(filename, delimiter=delim)

def LoadGT(fname,delim=','):
	gt=np.genfromtxt(fname, delimiter=delim)
	return gt
def LoadGroundTruth(fname,dims,idx=False):
	alpha=[];
	gt = np.zeros((dims,dims))
	i=0;
	offset=0
	if idx:
		offset=1
	#print ('Probing: ',fname);
	
	if not os.path.exists(fname):
		print ('Ground Truth file not found, nothing to compare');
		return gt,False;
				
	with open(fname,'r') as file:
		k = file.readlines();
		for i in range(len(k)):
			line_ = k[i].split('\t');
			s=int(line_[0].strip());
			t=int(line_[1].strip());
			gt[s-offset,t-offset]=1
			
	
	return gt,True;




def LoadPartialData(filename,N=1000):
	with open(filename,'r') as myfile:
		k = [next(myfile) for x in range(N)]

		dims = len(k[1].split(','));
		recs = len(k)-1;
		variables= np.zeros((1,dims));

		for i in range(1,recs):
			if 'nan' not in k[i].lower():
				line = k[i].split(',');
				temp=np.zeros((1,dims));

				for j in range(0,dims):
					temp[0,j]=line[j].strip();

				variables=np.vstack((variables,temp));
			else:
				recs=recs-1;
		variables=np.delete(variables,0,0);

	return variables;
