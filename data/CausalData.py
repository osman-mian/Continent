import cdt.data as tb
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(12)

def main():
	base_path="./data/"
	mechanisms=['polynomial']
	noises = ['uniform']
	node=[5,10,15]

	offset=0
	batch=15
	md=2
	for nd in node:
		for mec in mechanisms:
			for n in noises:
				print("Node: ",nd,": ",mec,", ",n)
				for i in range(batch):
					generator = tb.AcyclicGraphGenerator(mec,nodes=nd, npoints=10000, noise=n, noise_coeff=0.3,dag_type='erdos',expected_degree=md)
					df, G = generator.generate()
					data=df.to_numpy()
					graph=nx.to_numpy_array(G).astype(int)
					print(data.shape," : ",np.sum(graph))
					print(base_path+'data'+str(offset+i+1) +".txt")
					np.savetxt(base_path+'data'+str(offset+i+1) +".txt"	  , data , delimiter=',')   # X is an array
					np.savetxt(base_path+'data'+str(offset+i+1) +"_truth.txt", graph, delimiter=',',fmt='%d')
				offset+=batch

		


main();



'''
x=0
		counter=0
		while counter<100:
			counter+=1
			x= data[:,int(input("x:"))]
			y= data[:,int(input("y:"))]
			plt.scatter(x, y)	
			plt.show()
			#import ipdb;ipdb.set_trace()
'''
