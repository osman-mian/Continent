#os libs
from copy import deepcopy
import warnings

#scikit and numpy
import matplotlib.pyplot as plt


import numpy as np

#learners

from globe.globeWrapper import *

class Learner:

	def __init__(self,id_):
		self.is_init=True

	def learn(self,data):
		raise NotImplementedError

	def continue_learning(self,data,config):
		raise NotImplementedError

class Globe(Learner):
	def __init__(self,M=1):
		super().__init__(self)
		self.M = M
		self.glb = GlobeWrapper(M)

	def learn(self,data,config=None):
		self.glb.set_vars(data)

		if config is None:
			network,models,meta_data = self.glb.run()
		else:
			network,models,meta_data = self.glb.resume(config)

		
		return network,models,meta_data

	def learn_scm(self,data,network):
		return self.glb.learn_scm(data,network)

	def residue(self,X_,Y_true,model):
		mo=model['model']
		r_predict=mo[0]
		rearth=mo[1]
		Y_hat = self.glb.predict(X_,r_predict,rearth)
		diff=Y_hat.reshape(-1)-Y_true.reshape(-1)
		return diff.reshape((-1,1))

	def data_given_model_cost(self,X_,Y_true,model):
		mo=model['model']
		r_predict=mo[0]
		rearth=mo[1]

		Y_hat = self.glb.predict(X_,r_predict,rearth)
		#print(Y_hat)
		#print(Y_true)
		diff=Y_hat.reshape(-1)-Y_true.reshape(-1)
		sse = np.sum(diff**2)
		#print("Shape in eval: ",np.shape(diff))
		#print("SSE: ",np.round(sse,5))

		rows = Y_hat.shape[0]
		return self.glb.data_given_model_cost(sse,rows,Y_true)

	def predict(self,X,r_predict,rearth):
		return self.glb.predict(X,r_predict,rearth)

	def score(self,source,target,dim):
		return self.glb.score(source,target,dim)
		
