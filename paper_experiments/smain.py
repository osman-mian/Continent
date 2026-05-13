import warnings
with warnings.catch_warnings():
	warnings.simplefilter("ignore")#, category=UserWarning)
	import sys
	import os
	import utils as utils
	from copy import deepcopy
	import numpy as np
	import cdt
	import pickle
	import random
	from cdt.metrics import *
	from Methods import *
	np.random.seed(91)

	def main(id_,append):
		gt 	 	 = LoadGT('./data/'+append+'/data'+id_+'_truth.txt')
		methods  = [Continent()]#,GlobeX(),GESx(),JCIPC(),Resit(),Lingam(),DirectLingam()]
		tup		 = []
		for s_ind in range(1):
			filename = './data/'+append+'/data'+id_+'.txt'
			x = utils.LoadData(filename)
			dt_all = x[x[:, s_ind].argsort()]
			split_dt = np.array_split(dt_all,5)
			dlist=[]
			for sdata in split_dt:
				np.random.shuffle(sdata)
				dlist.extend(np.array_split(sdata,2))
			
			dims=dt_all.shape[1]
			random.shuffle(dlist)
			mods = len(dlist)

			for i in range(len(methods)):
				#print(methods[i].name)
				try:
					network,sds,f1s,mcount=methods[i].learn(dlist,gt)
					shd = SHD(gt, network, double_for_anticausal=False)
					tup.append( ( (filename,append,methods[i].name,sds,f1s,[len (mc) for mc in mcount],mods,dims),mcount) )
				except Exception as e: 
					print(e)
					print("something went wrong...")
					import ipdb;ipdb.set_trace()

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
		subfolders=["osc","poly"]
		fname="./results/biased.csv"
		pklname="./results/biased.pkl"

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
				
				with open(pklname, 'wb') as f:
					pickle.dump(res_list, f)

run()




