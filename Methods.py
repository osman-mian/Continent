import lingam
import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci

from utils import *
import networkx as nx
import numpy as np
import cdt

from Learner import Globe
from ConCausD import *
class Continent:
	def __init__(self):
		self.name="Continent";
		self.learner = ConCausD(Globe());

	def learn(self,dlist,gt):
		dims = dlist[-1].shape[1]
		envs = len(dlist)
		network = np.zeros((dims,dims))
		mcount=[];
		sds = [];
		f1s = [];
		ccd=self.learner
		for j in range(envs):
			print("file: ",(j+1)," of ", envs)
			data = dlist[j]
			ccd.update_model(data)
			networks = [m.network for m in ccd.models]
			mcount.append(deepcopy(networks))
			network = ccd.get_network();
			sds.append(int(cdt.metrics.SHD(gt, network, double_for_anticausal=False)))
			f1s.append(F1_Or(gt,network))

		ccd.attempt_merge()
		networks = [m.network for m in ccd.models]
		mcount.append(deepcopy(networks))
		network = ccd.get_network();

		shd = cdt.metrics.SHD(gt, network, double_for_anticausal=False)
		sds.append(shd)

		f1 = F1_Or(gt, network)
		f1s.append(f1)
		return network,sds,f1s,mcount


class GlobeX:
	def __init__(self):
		self.name="Globe";
		self.learner= Globe();

	def learn(self,dlist,gt):
		dims = dlist[-1].shape[1]
		envs = len(dlist)
		network = np.zeros((dims,dims))
		sds=[]
		f1s=[]
		mcount=[];
		for i in range(len(dlist)):
			data=Standardize(dlist[i])
			nx,_ = self.learner.learn(data)
			network+= np.array(np.array(nx,dtype=bool),dtype=np.int8)
			n2 = np.copy(network)
			n2[ n2 <  (i+1)/2  ] = 0
			n2[ n2 >= (i+1)/2  ] = 1
			sds.append(int(cdt.metrics.SHD(gt, n2, double_for_anticausal=False)))
			f1s.append(F1_Or(gt,n2))
			mcount.append([1])

		network[ network <  envs/2  ] = 0
		network[ network >= envs/2  ] = 1
		sds.append(int(cdt.metrics.SHD(gt, network, double_for_anticausal=False)))
		f1s.append(F1_Or(gt,network));
		mcount.append([network])
		return network,sds,f1s,mcount

class JCIPC:
	def __init__(self):
		self.name="JCI-PC"
		self.learner = pc

	def learn(self,dlist,gt=None):
		dims = dlist[-1].shape[1]
		
		envs = len(dlist)
		context = np.eye(envs)

		data=[]

		for i in range(envs):
			e_data = Standardize(dlist[i])
			context_col = context[i,:]
			context_data = np.tile(context_col,(e_data.shape[0],1))
			e_data = np.hstack( (context_data,e_data) )
			data.append(e_data)

		data 	= np.vstack(data)
		cg	= self.learner(data,0.05,uc_rule=1,verbose=False,show_progress=False)
		network = cg.G.graph

		network = cg.G.graph * 0;
		ex_graph= cg.G.graph[envs:data.shape[1],envs:data.shape[1]]
		network = network[envs:data.shape[1],envs:data.shape[1]]

		#cg : a CausalGraph object, where 
		#cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicate i –> j; 
		#cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicate i — j; 
		#cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.
		for i in range(dims):
			for j in range(i+1,dims):
				if ex_graph[j,i]==1 and ex_graph[i,j]==-1:
					network[i,j]=1
					network[j,i]=0
				if ex_graph[j,i]==-1 and ex_graph[i,j]==-1:
					network[i,j]=1
					network[j,i]=1
		sds=[1 for d in dlist]
		f1s=[0 for d in dlist]
		mcount=[[1] for d in dlist]
		sds.append(int(cdt.metrics.SHD(gt, network, double_for_anticausal=False)))
		f1s.append(F1_Or(gt,network))
		mcount.append([network])
		return network,sds,f1s,mcount


class GESx:
	def __init__(self):
		self.name="GES"
		self.learner = ges

	def learn(self,dlist,gt):
		dims = dlist[-1].shape[1]
		envs = len(dlist)
		network = np.zeros((dims,dims))
		sds=[]#cdt.metrics.SHD
		f1s=[]
		mcount=[];
		for i in range(len(dlist)):
			data=Standardize(dlist[i])
			net, _ = self.learner.fit_bic(data)
			network+= net
			n2 = np.copy(network)
			n2[ n2 <  (i+1)/2  ] = 0
			n2[ n2 >= (i+1)/2  ] = 1
			sds.append(int(cdt.metrics.SHD(gt, n2, double_for_anticausal=False)))
			f1s.append(F1_Or(gt,n2))
			mcount.append([1])

		network[ network <  envs/2  ] = 0
		network[ network >= envs/2  ] = 1
		sds.append(int(cdt.metrics.SHD(gt, network, double_for_anticausal=False)))
		f1s.append(F1_Or(gt,network))
		mcount.append([network])
		return network,sds,f1s,mcount


from sklearn.ensemble import RandomForestRegressor
class Resit:
	def __init__(self):
		self.name="RESIT"
		reg = RandomForestRegressor(max_depth=4, random_state=0)
		self.learner = lingam.RESIT(reg)

	def learn(self,dlist,gt):
		dims = dlist[-1].shape[1]
		envs = len(dlist)
		network = np.zeros((dims,dims))

		sds=[]
		f1s=[]
		mcount=[];
		for i in range(len(dlist)):
			data=Standardize(dlist[i])
			self.learner.fit(data)
			network+= np.array(np.array(self.learner.adjacency_matrix_,dtype=bool),dtype=np.int8)
			n2 = np.copy(network)
			n2[ n2 <  (i+1)/2  ] = 0
			n2[ n2 >= (i+1)/2  ] = 1
			sds.append(int(cdt.metrics.SHD(gt, n2, double_for_anticausal=False)))
			f1s.append(F1_Or(gt,n2))
			mcount.append([1])

		network[ network <  envs/2  ] = 0
		network[ network >= envs/2  ] = 1
		sds.append(int(cdt.metrics.SHD(gt, network, double_for_anticausal=False)))
		f1s.append(F1_Or(gt,network))
		mcount.append([network])
		return network,sds,f1s,mcount

class Lingam:
	def __init__(self):
		self.name="Lingam"
		self.learner = lingam.DirectLiNGAM()

	def learn(self,dlist,gt):
		dims = dlist[-1].shape[1]
		envs = len(dlist)
		network = np.zeros((dims,dims))
		sds=[]
		f1s=[]
		mcount=[];
		for i in range(len(dlist)):
			data=Standardize(dlist[i])
			self.learner.fit(data)
			network+= np.array(np.array(self.learner.adjacency_matrix_,dtype=bool),dtype=np.int8)
			n2 = np.copy(network)
			n2[ n2 <  (i+1)/2  ] = 0
			n2[ n2 >= (i+1)/2  ] = 1
			sds.append(int(cdt.metrics.SHD(gt, n2, double_for_anticausal=False)))
			f1s.append(F1_Or(gt,n2))
			mcount.append([1])

		network[ network <  envs/2  ] = 0
		network[ network >= envs/2  ] = 1
		sds.append(int(cdt.metrics.SHD(gt, network, double_for_anticausal=False)))
		f1s.append(F1_Or(gt,network));
		mcount.append([network])
		return network,sds,f1s,mcount



class DirectLingam:
	def __init__(self):
		self.learner=lingam.MultiGroupDirectLiNGAM();
		self.name="DirectLingam"

	def learn(self,dlist,gt=None):
		mods	 = len(dlist)
		for i in range(mods):
			dlist[i]=Standardize(dlist[i])

		self.learner.fit(dlist)
		sds=[1 for d in dlist]
		f1s=[0 for d in dlist]
		mcount=[[1] for d in dlist]
		network= np.array(np.array(self.learner.adjacency_matrix_,dtype=bool),dtype=np.int8)
		sds.append(int(cdt.metrics.SHD(gt, network, double_for_anticausal=False)))
		f1s.append(F1_Or(gt,network))
		mcount.append([network])
		return network,sds,f1s,mcount
