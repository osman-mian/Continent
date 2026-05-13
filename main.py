#######################################
#
#Author: osman ali mian
#Email: osmanalimian@live.com.pk
#Last Modified: 28.May.2024
#
#########################################

import utils as utils
import numpy as np
from Methods import *

def main():
	filename 	= './data/poly/data1.txt'		#data file
	gtname 		= './data/poly/data1_truth.txt'	#ground truth file 
	method  	= Continent() 					#see Methods.py for other competitor methods. Should be easy to load and run


	#Prepare datasets. Load a single file and split it into episodes
	dt_all   	= utils.LoadData(filename)			#load data
	gt 	 	 	= LoadGT(gtname)					#load Ground truth, if no ground truth file exists, you can replace this line and create any random matrice of size  dims x dims
	mods 	 	= 10								#number of episodes to split into
	dims 	 	= dt_all.shape[1]					#number of variables in data


	np.random.shuffle(dt_all)						#shuffle data
	dlist	 	= np.array_split(dt_all,mods)		#split into episodes

	try:
		#returns 
		#network = best compressing network if forced to choose 1 network (its better to use the mcount array instead for a fair comparison in this setting)
		#sds	 = the progression of hamming distance across increasing episodes
		#sis	 = the progression of intervention distance across increasing episodes
		#f1s	 = the progression of f1 scores over edge orientations across increasing episodes
		#mcount  = networks maintained across increasing episodes
		network,sds,sis,f1s,mcount=method.learn(dlist,gt)
		print("Network if forced to choose one: \n", network)
		print("SHD progress: ",sds)
		print("SID progress: ",sis)
		print("F1  progress: ",f1s)
		print("Model counts: ", [len(mc) for mc in mcount])
	except Exception as e: 
		print(e)
		print("something went wrong...")



	


main()




