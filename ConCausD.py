import numpy as np
from Model import Model
from Learner import *
from copy import deepcopy
from ResidualTests import *

class ConCausD:
	def __init__(self,learner,max_delay=3,rs = KolmogorovSmirnovTest()):
		self.learner = learner
		self.models=[];
		self.max_update_delay=3;
		self.tolerance=0
		self.sig_threshold=10
		self.residual_test=rs

	def evaluate_compressions(self,models,data,curr_model):
		curr_cost  	 = curr_model.compute()									#learn a model over new data

		samples 	 = [model.get_data().shape[0] for model in models]		
		curr_samples = np.sum(samples)										#sum up samples in current models
		tot_samples  = 1.0 * (curr_samples + data.shape[0])					#add samples of current episode to get the new total

		a0 = [model.self_evaluate() 				for model in models]	#each models performance on its own data
		a1 = [model.evaluate(data)  				for model in models]	#each models performance with current data
		a2 = [curr_model.evaluate(model.get_data())	for model in models]	#current models performance with each models data

		total 		= np.sum(a0)											#sum over current costs, for use in calculation in the for-loop
		new_cost 	= (total + curr_cost) / tot_samples						#We assume adding a new model is the best way to go so thats the best initial cost
		best_cost   = 9e99
		best_index 	= -1			
		best_diff 	= self.sig_threshold


		for i in range(len(models)):
			curr_arr 	= np.copy(a0)										#load all original costs
			min_val 	= np.min([a1[i],a2[i]])								#choose the smaller of alternate costs
			curr_arr[i] = min_val											#replace the smaller cost inside the original costs
			adj_sum 	= np.sum(curr_arr) / tot_samples					#normalize


			#print("samples: ",tot_samples)
			#print(adj_sum,": ",new_cost)
			#print(best_cost,",", adj_sum)

			if adj_sum <  best_cost:								
				best_cost 	= adj_sum										#store the best sum
				best_index	= i												#mark best index
				best_diff   = 0												#need the difference threshold only for the first time


		if  best_cost - new_cost > self.sig_threshold:			#no-hypercompression inequality, if storing separate model is more than alpha bits better, we introduce it
			best_index = -1
		#print(best_cost-new_cost)
		return best_index,a1,a2

	def update_model(self,data):

		self.tolerance+=1

		if len(self.models)==0:												#first model, nothing to compare
			curr_model = Model(self.learner,data)
			curr_model.compute()
			self.models.append(curr_model)
			return

		#do a residual test
		dims = data.shape[1]
		mods = len(self.models);

		pv_vector	= np.zeros((dims,mods))
		res_mat		= np.zeros((dims,mods))

		for i in range(len(self.models)):
			pv_vector[:,i],res_mat[:,i] = self.residual_test.test(data, self.models[i])

		#print(pv_vector)
		#print(np.sum(res_mat,axis=0))
		res_sum = np.nonzero(np.sum(res_mat,axis=0)==0)[0]
		#import ipdb;ipdb.set_trace()
		if res_sum.shape[0]!=0:
			best_index = res_sum[0]
			self.models[best_index].update_model(data)
			#print("No need to learn new model..")
		else:
			#print("Have to learn new model")
			curr_model = Model(self.learner,data)
			best_index,a1,a2 = self.evaluate_compressions(self.models,data,curr_model)

			if best_index == -1 :											#No better adjustment found
				#print("No better adjustment found")
				self.models.append(curr_model)
				#print(len(self.models))
			elif a1[best_index] <  a2[best_index]:							#One of the existing models was a better fit for the data
				#print("Merging data into existing model")
				self.models[best_index].update_model(data)
			elif a1[best_index] >= a2[best_index]:							#The new model was a better fit for some existing data
				#print("Merging existing model into new model")
				curr_model.update_model(self.models[best_index].get_data())
				self.models[best_index]=curr_model

		if self.tolerance==self.max_update_delay:							#Attempt to update parameters after certain number of steps
			#print("Attempting to merge after ",self.tolerance," steps")
			self.tolerance=0
			self.update_params();											#continue learning local models from where we left off..
			self.attempt_merge();											#try and merge once done updating


	def update_params(self):
		for model in self.models:											#re-learn the networks and SCM for existing models
			model.compute()

	def attempt_merge(self):
		m=len(self.models)
		i=0;
		if m==1:
			return #Nothing to merge

		while i<m:
			pairs = [(i,j) for j in range(i+1,m)]
			best_diff=self.sig_threshold
			best_model=None
			a0 = [model.self_evaluate() 	for model in self.models]	#calculate bits for the current configuration (null model)
			total_samples = np.sum([model.get_data().shape[0] for model in self.models])
			t0 = np.sum(a0)/total_samples
			best_case = 9e99

			for pair in pairs:
				m1		= self.models[pair[0]]														#get the merging candidates
				m2		= self.models[pair[1]]
				d12		= np.vstack( (m1.get_data(),m2.get_data()) )								#merge the data of both models
				m12_1	= Model(self.learner,d12,m1.network,m1.model,m1.config)						#create a new model over combined data but using structure from mod1
				m12_2	= Model(self.learner,d12,m2.network,m2.model,m2.config)						#create a new model over combined data but using structure from mod2

				m12_1.update_scm()
				m12_2.update_scm()
				#import ipdb;ipdb.set_trace()
				m1_test = self.residual_test.test_samples(m1.get_data(), m1,m12_1) and self.residual_test.test_samples(m2.get_data(), m2,m12_1)
				m2_test = self.residual_test.test_samples(m1.get_data(), m1,m12_2) and self.residual_test.test_samples(m2.get_data(), m2,m12_2)

				if m1_test:
					print("Combined scm works!")
					m12=m12_1
				elif m2_test:
					print("Combined scm works!")
					m12=m12_2
				else:
					print("Combined scm breaks")
					continue

				trunc_model=[self.models[k] for k in range(m) if (k!=pair[0] and k!=pair[1])]	#create a new model list with other remaining models
				trunc_model[i:i]=[m12]															#add joint model to this to complete the configuration, (added always to index i for implementation purpose)
				a12 =[model.self_evaluate() 	for model in trunc_model]						#calculate bits for the alternate configuration
				t12 = np.sum(a12)/total_samples

				if t12<best_case:
					best_case=t12
					best_model=trunc_model

			if best_model is not None and t12 - t0 > self.sig_threshold:
				best_model=None

			if best_model is not None:
				self.models=best_model
				#print("merge happend")
				i=i-1

			i=i+1
			m=len(self.models)
			


	def attempt_merge1234(self):
		m=len(self.models)
		i=0;
		if m==1:
			return #Nothing to merge

		while i<m:
			pairs = [(i,j) for j in range(i+1,m)]
			best_diff=self.sig_threshold
			best_model=None
			a0 = [model.self_evaluate() 	for model in self.models]	#calculate bits for the current configuration (null model)
			total_samples = np.sum([model.get_data().shape[0] for model in self.models])
			t0 = np.sum(a0)/total_samples
			best_case = 9e99
			for pair in pairs:
				m1		= self.models[pair[0]]							#get the models to merge
				m2		= self.models[pair[1]]
				d12		= np.vstack( (m1.get_data(),m2.get_data()) )	#merge the data of both models
				m12		= Model(self.learner,d12)						#create a new model over combined data
				bits12	= m12.compute()									#learn the model

				trunc_model=[self.models[k] for k in range(m) if (k!=pair[0] and k!=pair[1])]	#create a new model list with other remaining models
				trunc_model[i:i]=[m12]															#add joint model to this to complete the configuration, (added always to index i for implementation purpose)
				a12 =[model.self_evaluate() 	for model in trunc_model]						#calculate bits for the alternate configuration
				t12 = np.sum(a12)/total_samples

				if t12<best_case:
					best_case=t12
					best_model=trunc_model

			if t12 - t0 > self.sig_threshold:
				best_model=None

			if best_model is not None:
				self.models=best_model
				#print("merge happend")
				i=i-1

			i=i+1
			m=len(self.models)
			

	def get_network(self):
		envs=len(self.models)
		if envs==1:
			return self.models[0].network

		network = self.models[0].network*0
		for i in range(envs):
			network+= self.models[i].network

		dt 		 = np.vstack([model.get_data() 						for model in self.models])
		best_idx = np.argmin([model.evaluate(dt,use_local=False)	for model in self.models])
		return self.models[best_idx].network
		

