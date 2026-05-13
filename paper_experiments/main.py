#os libs
import warnings

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	import sys
	import os
	import utils as utils
	from copy import deepcopy
	import numpy as np
	import cdt
	import pickle
	from cdt.metrics import *
	from Methods import *
	np.random.seed(91)

	def main(id_,append):
		filename = './data/'+append+'/data'+id_+'.txt'
		dt_all   = utils.LoadData(filename)
		np.random.shuffle(dt_all)
		gt 	 	 = LoadGT('./data/'+append+'/data'+id_+'_truth.txt')
		mods 	 = 10
		dims 	 = dt_all.shape[1]
		dlist	 = np.array_split(dt_all,mods)

		tup		 = []
		methods  = [Continent()]#,GlobeX(),GESx(),JCIPC(),Lingam(),DirectLingam(),Resit()]


		for i in range(len(methods)):
			#print(methods[i].name)
			try:
				network,sds,f1s,mcount=methods[i].learn(dlist,gt)
				shd = SHD(gt, network, double_for_anticausal=False)
				tup.append( ( (filename,append,methods[i].name,sds,f1s,[len (mc) for mc in mcount],mods,dims),mcount) )
			except Exception as e: 
				print(e)
				print("something went wrong...")

		return tup

		
	def to_string(t_list):
		init = ""
		from datetime import datetime
		for tt in t_list:
			for t in tt[0]:
				init = init + str(t).replace('[','').replace(']','')+","
			init+=str(datetime.now())+"\n"
		print(init)
		return init


	def run():
		subfolders=["poly","osc"]
		fname="./results/iid.csv"
		pklname="./results/iid.pkl"

		start_=0;
		end_=45;
		pre_load=True
		save=True


		if len(sys.argv) > 1 : subfolders 	= [subfolders[int(sys.argv[1])]];	#only run it for specific data type
		if len(sys.argv) > 2 : start_	 	= int(sys.argv[2]);
		if len(sys.argv) > 3 : end_ 		= int(sys.argv[3]);
		if len(sys.argv) > 4 : pre_load 	= bool(sys.argv[4]);
		if len(sys.argv) > 5 : save 		= bool(sys.argv[5]);

		print(fname)

		with open(fname,"a") as myfile:
			myfile.write("fname,data,method,Shd1,2,3,4,5,6,7,8,9,10,11,Fs1,2,3,4,5,6,7,8,9,10,11,Mods1,2,3,4,5,6,7,8,9,10,11,Total_models,vars,Time\n")

		for append in subfolders:
			print(append)
			res_list= []
			if pre_load and os.path.exists(pklname):
				with open(pklname, 'rb') as f:
					res_list=pickle.load(f)

			for i in range(start_,end_):
				exp_outcome =main(str(i),append)
				res_list.append(exp_outcome)

				with open(fname,"a") as myfile:
					myfile.write(to_string(exp_outcome))
				
				if save:
					with open(pklname, 'wb') as f:
						pickle.dump(res_list, f)

run()




