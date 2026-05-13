import pickle
import sys
from copy import deepcopy
import numpy as np
def main():
	fname="./results_osc/new_res_"+sys.argv[1]+'.pkl'
	print(fname)

	res_list=[];
	with open(fname, 'rb') as f:
		res_list = pickle.load(f)

	print(len(res_list))
	z =[]
	del res_list[0]
	for tup in res_list:
		nets = tup[1]
		val = [len(n) for n in nets]
		z.append(deepcopy(val))
		#print(val)

	do_math(z)

def smain():
	fname="./results/res_new_sb"+sys.argv[1]+'.pkl'
	print(fname)

	res_list=[];
	with open(fname, 'rb') as f:
		res_list = pickle.load(f)

	#print(len(res_list))
	z =[]

	for tup in res_list:
		for sub_tup in tup:
			nets = sub_tup[1]
			val = [len(n) for n in nets]
			z.append(deepcopy(val))
			#print(val)

	#print(len(z))
	do_math(z,3)

def do_math(z,n=1):
	z=np.vstack(z)
	#print(z.shape[0]/n)
	if z.shape[0]>=25*n:
		z5   = np.average(z[0:25*n,:],axis=0)
		print("05: ",np.round(z5,2))

	if z.shape[0]>=50*n:
		z10  = np.average(z[25*n:50*n,:],axis=0)
		print("10: ",np.round(z10,2))

	if z.shape[0]==73*n:
		z15	 = np.average(z[50*n:73*n,:],axis=0)
		print("15: ",np.round(z15,2))
	

	z    = np.average(z,axis=0)
	print("To: ",np.round(z,2))



main()
#smain()
