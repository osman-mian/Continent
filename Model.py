import numpy as np
from copy import deepcopy
from utils import *
class Model:

	def __init__(self,learner,data,ntw=None,mod=None,cfg=None):
		self.data=[data];
		self.learner=learner
		self.model=mod
		self.network=ntw
		self.config=cfg
		self.dims=data.shape[1]
		self.needs_update=True
		self.bits=0
		self.stale=True
		self.last_eval=-9999

	def compute(self):
		if not self.needs_update:
			#print("Same model as before..")
			return self.bits

		#print("computing..")
		adt=self.data[0]
		if len(self.data)>1:
			adt = np.vstack(self.data)
		dt=Standardize(adt)
		#dt=adt
		n,m,md 		 = self.learner.learn(dt,self.config);
		self.model	 = m
		self.network = n
		self.config	 = md
		total_cost	 = 0

		#print(self.network)
		for i in range(self.dims):
			total_cost = total_cost + self.model[i]['lm']+self.model[i]['ldm']

		self.bits=total_cost[0]#/dt.shape[0]
		self.needs_update=False
		return self.bits

	def evaluate(self,dt,use_local=True):
		total_cost= 0
		network = self.network
		if use_local:
			self.data.append(dt)
			data = np.vstack(self.data)
		else:
			data = dt

		data =Standardize(data)
		for i in range(self.dims):
			cols=[]
			Y_true = np.copy(data[:,i].reshape(-1,1))
			cols.append((Y_true**0).reshape(-1,1))
			idx = np.argwhere(network[:,i]==1)

			if len(idx)>0:
				idx=idx[0]

			for id_ in idx:
				cols.append(data[:,id_].reshape(-1,1))

			X_ = np.hstack(cols)
			ldm = self.learner.data_given_model_cost(X_,Y_true,self.model[i])
			lm = self.model[i]['lm']
			total_cost =  total_cost + lm + ldm

			
		bits = total_cost[0] #/ data.shape[0]

		if use_local:
			del self.data[-1]
		return bits


	def get_residue(self,data):
		network = self.network
		data =Standardize(data)
		res=[]
		for i in range(self.dims):
			cols=[]
			Y_true = np.copy(data[:,i].reshape(-1,1))
			cols.append((Y_true**0).reshape(-1,1))
			idx = np.argwhere(network[:,i]==1)

			if len(idx)>0:
				idx=idx[0]

			for id_ in idx:
				cols.append(data[:,id_].reshape(-1,1))

			X_ = np.hstack(cols)
			res.append(self.learner.residue(X_,Y_true,self.model[i]))

		return np.hstack(res)

	def model_cost(self):
		lm=0
		for i in range(self.dims):
			lm += self.model[i]['lm']
		return lm
	

	def residual_eval(self,data):
		model_cost=self.model_cost()
		total_cost =self.evaluate(data)
		res_cost = (total_cost - model_cost)
		return res_cost

	def self_residual_eval(self):
		model_cost=self.model_cost()
		total_cost =self.evaluate(self.get_data(),False)
		res_cost = (total_cost - model_cost)
		return res_cost

	def self_evaluate(self):
		if self.stale:
			self.stale=False
			self.last_eval=self.evaluate(self.get_data(),False)

		return self.last_eval

	def update_model(self,dt):
		self.data.append(dt)
		self.stale=True
		self.needs_update=True


	def update_scm(self):
		self.model = self.learner.learn_scm(self.get_data(),self.network)
			

	def get_data(self):
		return np.vstack(self.data)

		
'''
def main():
	x = np.zeros((5,10))
	miss = set([0,2,9])
	x[:,list(miss)]=np.nan
	
	env = Environment(x,miss)


main()
#'''
