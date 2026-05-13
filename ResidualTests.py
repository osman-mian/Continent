from scipy.stats import ks_2samp
import numpy as np
class KolmogorovSmirnovTest():

	def __init__(self,alph=0.001):
		self.alpha = alph

	def ks_test(self,v1,v2):
		if v2.shape[1]!=v1.shape[1]:
			raise Exception("Error Data and model shapes do not match")

		dims = v1.shape[1]
		p_vector = np.zeros(dims)
		r_vector = np.zeros(dims)
		#print(v2.shape[0]," and ",v1.shape[0])
		for i in range(dims):
			sample1 = v1[:,i]
			sample2 = v2[:,i]
			sample1 = sample1 / np.std(sample1)
			sample2 = sample2 / np.std(sample2)

			_,p_vector[i] = ks_2samp(sample1, sample2)
			r_vector[i] = p_vector[i] < self.alpha

		return (p_vector,r_vector)

	def test(self,data,model):
		v1	= model.get_residue(data)
		v2  = model.get_residue(model.get_data())
		return self.ks_test(v1,v2)


	def test_samples(self,data,model1,model2):
		v1	= model1.get_residue(data)
		v2  = model2.get_residue(data)
		#import ipdb;ipdb.set_trace()
		p_vector,r_vector = self.ks_test(v1,v2)
		print(r_vector)
		#import ipdb;ipdb.set_trace()
		return np.sum(r_vector)==0




