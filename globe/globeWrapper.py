from globe.node import Node;
from globe.edge import Edge;
from globe.slope import Slope;
from globe.globe import Globe;
from globe.utils import *
from globe.dag import DAG;
from globe.statsCalculator import StatsCalculator
from globe.skeletonHandler import SkeletonHandler;
from globe.logger import Logger
from globe.RFunctions import *
import numpy as np;
import sys
import time as time
import os;
import gc;
from datetime import datetime

class GlobeWrapper:

	def __init__ (self,max_int,log_results=False,vrb=False):
		self.vars=np.zeros((5,5));
		self.M=max_int;
		self.log_path="./logs/log_"+ str(datetime.now(tz=None)).replace(' ','_')+ ".txt";
		self.log_flag= log_results;
		self.verbose=vrb;
		self.filename="";
		if self.log_flag:
			print("Saving results to: ",self.log_path)

	def predict(self,X,r_predict,rearth):
		return make_predict(X,r_predict,rearth)


	def data_given_model_cost(self,sse,rows,Y):
		mdiff = MinDiff(Y)
		ldm=Slope().gaussian_score_emp_sse(sse,rows,mdiff)
		return ldm

	def score(self,source,target,dim):
		rows = source.shape[0]
		k = source.shape[1]
		k=k-1 if k>1 else k
		mdiff = MinDiff(target)


		slope_ = Slope();
		globe_ = Globe(slope_,dims=dim,M=self.M);
		normalized_score,_ = globe_.ComputeScore(source,target,rows,mdiff,[k],True) 
		normalized_score = (normalized_score * 1.0) /rows
		return normalized_score


	def loadData(self,filename):
		with open(filename,'r') as file:
			k = file.readlines();

			dims = len(k[1].split(','));
			recs = len(k)-1;
			#dt = np.dtype('Float64')
			variables= np.zeros((1,dims));


			for i in range(1,recs):
				if 'nan' not in k[i].lower():
					line = k[i].split(',');
					temp=np.zeros((1,dims));

					for j in range(0,dims):
						temp[0,j]=line[j].strip();

					variables=np.vstack((variables,temp));
				else:
					recs=recs-1;
			variables=np.delete(variables,0,0);
		self.filename=filename;
		self.vars=variables;

	def set_vars(self,vars_):
		self.vars=vars_

	def resume(self,config):
		print("Resuming learning")
		Nodes			= config[0]
		Final_graph		= config[1]

		normalized_vars	= self.vars
		recs 			= normalized_vars.shape[0];
		dim 			= normalized_vars.shape[1];
		headers			= [i for i in range(0,dim)];
		inclusive_model	= True;
		point_model 	= True;

		logger = Logger(self.log_path,log_to_disk=self.log_flag,verbose=self.verbose);
		logger.Begin();

		slope_ = Slope();
		globe_ = Globe(slope_,dims=dim,M=self.M);
		Edges = [[None for x in range(dim)] for y in range(dim)];
		
		for i in range(dim):
			for j in range(dim):
				e1 = Edge(-1,[],np.array([0]),0);
				Edges[i][j]=e1

		sh = SkeletonHandler(slope_,globe_,logger);

		#here the data should be updated
		Nodes = [];
		for i in range(0,dim):	
			Nodes.append(Node(normalized_vars[:,i].reshape(recs,-1),globe_));


		for i in range(0,dim):	
			new_bits 	 = sh.UpdateNodeBits(Final_graph,Nodes,i)
			Nodes[i].SetCurrentBits(new_bits)
		

		graph_ = DAG(globe_,Nodes,Final_graph,Edges,headers,logger);
		graph_.BackwardSearch();

		for i in range(dim):
			row=[]
			for j in range(dim):
				if Final_graph[i][j] is not None:
					row.append("1")
				else:
					row.append("0")
			print(row)


		#import ipdb;ipdb.set_trace()
		#creating only those candidate edges that are not in final graph
		undirected_edges=[];
		for i in range(0,dim):
			for j in range(i+1,dim):
				if Final_graph[i][j] is None and Final_graph[j][i] is None:
					undirected_edges.append((i,j));


		#rescore the candidates
		#import ipdb;ipdb.set_trace()
		sh = SkeletonHandler(slope_,globe_,logger);
		pq,ri =sh.RerankEdges(undirected_edges,Nodes,Edges,Final_graph);	
		gc.collect();

		#prepare the configuration
		graph_.priority_queue	= pq
		graph_.reverse_index	= ri
		graph_.Edges			= Edges
		#import ipdb;ipdb.set_trace()
		#run the rest like usual
		#print("------------")
		#for i in range(dim):
		#	row=[]
		#	for j in range(dim):
		#		if Final_graph[i][j] is not None:
		#			row.append("1")
		#		else:
		#			row.append("0")
		#	print(row)

		graph_.BackwardSearch();
		gc.collect();
		#print("------------")
		#for i in range(dim):
		#	row=[]
		#	for j in range(dim):
		#		if Final_graph[i][j] is not None:
		#			row.append("1")
		#		else:
		#			row.append("0")
		#	print(row)
		#import ipdb;ipdb.set_trace()
		logger.WriteLog("END LOGGING FOR FILE: "+self.filename);
		logger.End();
		#import ipdb;ipdb.set_trace()
		network = np.zeros((dim,dim));


		for i in range(0,dim):
			for j in range(0,dim):
				if Final_graph[i][j] is not None:
					network[j][i] =1;	#need to flip indices (i,j to j,i) here because GLobe stores adjaceny matrix from Child to Parent
		#print(network)


		models ={};
		for i in range(dim):
			cols=[]
			Y_ = Nodes[i].GetData()
			cols.append((Y_**0).reshape(-1,1))
			#print("************Learning prediction for ",i)
			idx = np.argwhere(network[:,i]==1)
			if len(idx)>0:
				idx=idx[0]
			#print("Parents are")
			for id_ in idx:
				#print(id_)
				cols.append(Nodes[id_].GetData().reshape(-1,1))
			#print("---")
			X_ = np.hstack(cols)
			#print(X_.shape)

			rows = Y_.shape[0]
			k= [max(X_.shape[1]-1,1)]
			mindiff = Nodes[i].GetMinDiff()
			models[i]= globe_.ComputeModelScore(X_,Y_,rows,mindiff,k)

		#print(network)						
		meta_data = (Nodes,Final_graph)
		return network,models, meta_data


	def run(self):
		normalized_vars=self.vars#Standardize(self.vars);
		recs = normalized_vars.shape[0];
		dim = normalized_vars.shape[1];
		headers=[i for i in range(0,dim)];
		inclusive_model=True;
		point_model = True;

		slope_ = Slope();
		globe_ = Globe(slope_,dims=dim,M=self.M);

		logger = Logger(self.log_path,log_to_disk=self.log_flag,verbose=self.verbose);
		logger.Begin();
		logger.WriteLog("BEGIN LOGGING FOR FILE: "+self.filename);
		Edges = [[None for x in range(dim)] for y in range(dim)];
		Final_graph = [[None for x in range(dim)] for y in range(dim)];
		for k in range(dim):
			for j in range(dim):
				Edges[k][j]=Edge(-1,[],[],0);

		Nodes = [];
		for i in range(0,dim):	
			Nodes.append(Node(normalized_vars[:,i].reshape(recs,-1),globe_));

		undirected_edges=[];
		for i in range(0,dim):
			for j in range(i+1,dim):
				undirected_edges.append((i,j));
				#Plot2d(source,target,fname,save=False,rejected=False):
				#Plot2d(Nodes[i].GetData(),Nodes[j].GetData(),"./plots/visual_"+str(i)+"-"+str(j)+".png",True,False)
		#import ipdb;ipdb.set_trace()

		#*******************************************************************#
		sh = SkeletonHandler(slope_,globe_,logger);
		pq,ri =sh.RankEdges(undirected_edges,Nodes,Edges,Final_graph);	
		gc.collect();
		#import ipdb;ipdb.set_trace()
		graph_ = DAG(globe_,Nodes,Final_graph,Edges,headers,logger,pq,ri);
		graph_.ForwardSearch();
		graph_.BackwardSearch();
		gc.collect();
		logger.WriteLog("END LOGGING FOR FILE: "+self.filename);
		logger.End();
		#import ipdb;ipdb.set_trace()
		network = np.zeros((dim,dim));


		for i in range(0,dim):
			for j in range(0,dim):
				if Final_graph[i][j] is not None:
					network[j][i] =1;	#need to flip indices (i,j to j,i) here because GLobe stores adjaceny matrix from Child to Parent
		#print(network)


		models ={};
		for i in range(dim):
			cols=[]
			Y_ = Nodes[i].GetData()
			cols.append((Y_**0).reshape(-1,1))
			#print("************Learning prediction for ",i)
			idx = np.argwhere(network[:,i]==1)
			if len(idx)>0:
				idx=idx[0]
			#print("Parents are")
			for id_ in idx:
				#print(id_)
				cols.append(Nodes[id_].GetData().reshape(-1,1))
			#print("---")
			X_ = np.hstack(cols)
			#print(X_.shape)

			rows = Y_.shape[0]
			k= [max(X_.shape[1]-1,1)]
			mindiff = Nodes[i].GetMinDiff()
			models[i]= globe_.ComputeModelScore(X_,Y_,rows,mindiff,k)
						
		meta_data = (Nodes,Final_graph)
		return network,models,meta_data
