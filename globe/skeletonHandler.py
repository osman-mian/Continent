from globe.edge import Edge;
from queue import PriorityQueue
import numpy as np;
from globe.edge_ranking import Scorer;
import gc;


class SkeletonHandler:

	def __init__(self,slp,glb,lg):
		self.slope_=slp;
		self.globe_=glb;
		self.logger=lg;
		self.q = PriorityQueue();
		
	def UpdateNodeBits(self,Final_graph,Nodes,target_node):
		current_parents=[i for i,x in enumerate(Final_graph[target_node]) if x is not None];
		current_edges= [];
		parent_nodes=[];
		for current_parent in current_parents:
			current_edges.append(Final_graph[target_node][current_parent]);
			parent_nodes.append(Nodes[current_parent]);
		
		child = Nodes[target_node]
		gain_in_bits,new_bits,coeff,absolute_gain_in_bits=self.globe_.GetCombinationCost(parent_nodes,current_edges,child);
		return new_bits


	def RerankEdges(self,undirected_edges,Nodes,Edges,Final_graph):
		self.q = PriorityQueue();
		reverse_index = {};		
		cc=0;
		#print(undirected_edges)
		#import ipdb;ipdb.set_trace()
		#go through each entry
		for edge in undirected_edges:
			gc.collect()
			i = edge[0];
			j = edge[1];

			pa_i = [pas for pas,x in enumerate(Final_graph[i]) if x is not None];	#populate already existing parents
			ce_i = [];
			pa_i_nodes=[];
		
			for current_parent in pa_i:
				ce_i.append(Final_graph[i][current_parent]);
				pa_i_nodes.append(Nodes[current_parent]);

			gain_ratio1,best_new_bits1,x1_best_absolute1,best_fids1,coeffs1=self.globe_.GetEdgeAdditionCost(pa_i_nodes,Nodes[j],Nodes[i],ce_i);


			pa_j = [pas for pas,x in enumerate(Final_graph[j]) if x is not None];	#populate already existing parents
			ce_j = [];
			pa_j_nodes=[];
			for current_parent in pa_j:
				ce_j.append(Final_graph[j][current_parent]);
				pa_j_nodes.append(Nodes[current_parent]);

			gain_ratio2,best_new_bits2,x2_best_absolute2,best_fids2,coeffs2=self.globe_.GetEdgeAdditionCost(pa_j_nodes,Nodes[i],Nodes[j],ce_j);


			S = np.abs(x1_best_absolute1-x2_best_absolute2)
			score_ = -(S)
			
			self.logger.WriteLog("Ranked Edge between Nodes: "+str(i)+" and "+str(j));
			
			if x1_best_absolute1 > x2_best_absolute2:
				v1=score_;
				v2=-score_;
			else:
				v1=-score_;
				v2=score_;
			
			self.logger.WriteLog(str(gain_ratio1)+": "+str(x1_best_absolute1)+","+str(x2_best_absolute2));
			e1 = Edge(best_fids1,coeffs1,x1_best_absolute1,v1);
			e2 = Edge(best_fids2,coeffs2,x2_best_absolute2,v2);

			Edges[i][j] = e1;
			reverse_index[(i,j)] = (gain_ratio1,v1,x1_best_absolute1)
			if gain_ratio1 < 1:
				self.logger.WriteLog("Added to q: "+str(v1)+" : "+str(best_new_bits1));
				self.q.put( (v1,(i,j,best_new_bits1)));
			
			Edges[j][i] = e2;
			reverse_index[(j,i)] = (gain_ratio2,v2,x2_best_absolute2);
			if gain_ratio2<1:
				self.logger.WriteLog("Added to q: "+str(v2)+" : "+str(best_new_bits2));
				self.q.put((v2,(j,i,best_new_bits2)));

		return self.q,reverse_index;

	def RankEdges(self,undirected_edges,Nodes,Edges,Final_Graph):
		#The reverse index is to reverse reference the entries of the priority queue. This will help us later in checking if the priority queue entries are stale
		reverse_index = {};		
		cc=0;

		#go through each entry
		for edge in undirected_edges:
			gc.collect()
			i = edge[0];
			j = edge[1];

			#Rank edge in both possible directions
			gain_ratio1,best_new_bits1,x1_best_absolute1,best_fids1,coeffs1=self.globe_.GetEdgeAdditionCost([],Nodes[j],Nodes[i],[]);
			gain_ratio2,best_new_bits2,x2_best_absolute2,best_fids2,coeffs2=self.globe_.GetEdgeAdditionCost([],Nodes[i],Nodes[j],[]);
			
			#x1_best_absolute is analogous to the delta function described in the Algorithm section
			#x2_best_absolute is analogous to the delta function described in the Algorithm section
			#S is the PSI function described in the Algorithm section
			S = np.abs(x1_best_absolute1-x2_best_absolute2)
			score_ = -(S)
			
			self.logger.WriteLog("Ranked Edge between Nodes: "+str(i)+" and "+str(j));
			
			if x1_best_absolute1 > x2_best_absolute2:
				v1=score_;
				v2=-score_;
			else:
				v1=-score_;
				v2=score_;
			
			self.logger.WriteLog(str(gain_ratio1)+": "+str(x1_best_absolute1)+","+str(x2_best_absolute2));
			e1 = Edge(best_fids1,coeffs1,x1_best_absolute1,v1);
			e2 = Edge(best_fids2,coeffs2,x2_best_absolute2,v2);

			Edges[i][j] = e1;
			reverse_index[(i,j)] = (gain_ratio1,v1,x1_best_absolute1)
			if gain_ratio1 < 1:
				self.logger.WriteLog("Added to q: "+str(v1)+" : "+str(best_new_bits1));
				self.q.put( (v1,(i,j,best_new_bits1)));
			
			Edges[j][i] = e2;
			reverse_index[(j,i)] = (gain_ratio2,v2,x2_best_absolute2);
			if gain_ratio2<1:
				self.logger.WriteLog("Added to q: "+str(v2)+" : "+str(best_new_bits2));
				self.q.put((v2,(j,i,best_new_bits2)));

		
		return self.q,reverse_index;

		
	
			
		
		
