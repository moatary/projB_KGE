import parser
from makeDataReady2 import selectBestClusterIndices4EntitiesOverBenchmark ,selectBestClusterIndices4RelationsOverBenchmark

class batchloader():
	def __init__(self,args,mode='train'):
		import numpy as np
		self.batchsize=args.batchsize
		self.Valid_batchsize=args.validbatchsize
		self.Test_batchsize=args.testbatchsize
		self.samplertype=args.batchsampler
		# self.currenthead=0
		# self.currentrel=0
		self.maxiter=args.maxiter
		self.Test_maxiter=args.testmaxiter
		self.Valid_maxiter=args.validmaxiter
		#
		self.trainsegment=args.trainsegment
		self.validsegment=args.validsegment
		self.testsegment= args.testsegment
		self.setmode(mode)
		self._nonzeroheadrelsinds={}
		#
		# import sumovertails matrix to see where should sample
		self.sumovertail= args.tails_glossary_tensor.sumover('tails')
		if self.samplertype=='shuffle':
			# define indices list using shared memory:
			self.indices=[]#indicesperiter=[[]]*self.maxiter
			# binarize sumovertails matrix
			self.sumovertail= self.sumovertail>0
			self.hlen, self.rlen= args.tails_glossary_tensor.hlen, args.tails_glossary_tensor.rlen
			# self.sumovertail.reshape(1,-1)
			# find all indices of nonzero positions of sumovertails matrix:
			self.nonzeroheadrelsinds = self.sumovertail.nonzero()# self.indicestmp
			self.eachweight = np.ones(self.sumovertail.shape) * self.maxiter
			pass
			#
			# not needed self.sumovertail anymore, so get rid of it:
			# del self.sumovertail
			# define shuffle vector and pop it each time one (head,rel) used. pool samples out c(2,n)
		#TODO1: DON'T FORGET TO TUNE NEGSAMPLERTHRESH
		if self.samplertype=='weighted': # use sumovertails in normalized way
			# generate vector with each weight of sumovertails divided by avg produced by batchsize , then np.floor
			self.eachweight = np.floor(self.sumovertail*self.maxiter/(np.sum(self.sumovertail)/np.sum(self.sumovertail>0)))
			# find nozeros of the vector , rearrange main vector by seeked indices
			self.nonzeroheadrelsinds= self.sumovertail.nonzero()
			del self.sumovertail
		if self.samplertype=='adaptive':
			self.items_count_to_keep_track= np.ceil(self.batchsize/2) #todo: tune
			self.items_with_least_validscores_indices= np.arange(0,self.items_count_to_keep_track)
			self.items_with_least_validscores_score = 1000 #self.items_with_least_validscores_scores= 1000*np.ones((self.items_count_to_keep_track))
			self.items_with_most_validscores_indices= np.arange(self.items_count_to_keep_track, 2*self.items_count_to_keep_track)
			self.items_with_most_validscores_score = -1000#self.items_with_most_validscores_scores= -1000*np.ones((self.items_count_to_keep_track))
			self.minvalidscore, self.maxvalidscore, self.minvalidscoreindex, self.maxvalidscoreindex = 0, 0, 0, 0
			# find nonzeros of sumovertail metric and associate a uniform vector each with num of batchsize
			self.nonzeroheadrelsinds = self.sumovertail.nonzero()
			self.eachweight= np.ones(self.sumovertail.shape)*self.maxiter
			# del self.sumovertail
		#######
		self.indices = {}
		#
		# now decide and segment for train valid test:
		indices2seg = np.random.permutation(len(self.nonzeroheadrelsinds[0]))
		self.indices4seg = {}
		self.indices4seg['train'] = indices2seg[:int(np.floor(self.trainsegment * len(indices2seg)))]
		self.indices4seg['valid'] = indices2seg[int(np.floor(self.trainsegment * len(indices2seg))):int(np.floor(self.trainsegment * len(indices2seg) + self.validsegment * len(indices2seg)))]
		self.indices4seg['test'] = indices2seg[int(np.floor(self.trainsegment * len(indices2seg) + self.validsegment * len(indices2seg))):]
		self._nonzeroheadrelsinds['train'] = (self.nonzeroheadrelsinds[0][self.indices4seg['train']], self.nonzeroheadrelsinds[1][self.indices4seg['train']])
		self._nonzeroheadrelsinds['valid'] = (self.nonzeroheadrelsinds[0][self.indices4seg['valid']], self.nonzeroheadrelsinds[1][self.indices4seg['valid']])
		self._nonzeroheadrelsinds['test' ] = (self.nonzeroheadrelsinds[0][self.indices4seg['test']], self.nonzeroheadrelsinds[1][self.indices4seg['test']])
		self.nonzeroheadrelsinds=self._nonzeroheadrelsinds

	def setmode(self,mode):
		if mode=='train' or mode=='training':
			self.trainmode=True
			self.validmode=False
			self.testmode=False
			self.mode='train'
		elif mode=='valid' or mode=='validation':
			self.trainmode=False
			self.validmode=True
			self.testmode=False
			self.mode='valid'
		elif mode=='test' or mode=='testing':
			self.trainmode=False
			self.validmode=False
			self.testmode=True
			self.mode='test'



	def update_eachweight(self, average_batch_loss_vec, new_validation_score ): # only for adaptive mode for attention mechanism
		##TODO: CONTAINS ERR:NOT SET FOR TRAIN VALID TEST
		import numpy as np
		# average_batch_score_vec: score for each headrel in batch which is averaged over tails (in the case it has not)
		# new_validation_score : score from validation set
		#self.currentlysampledindices
		# first find batch items with least average_batch_loss_vec. They are more responsible for the score_decrease / / / self.items_count_to_keep_track
		indices= self.currentlysampledindices[np.argsort(average_batch_loss_vec)[:self.items_count_to_keep_track]]
		if new_validation_score<self.items_with_least_validscores_score:
			# increase weights of 'most' and decrease 'current'
			self.eachweight[indices]-=1
			self.eachweight[self.items_with_most_validscores_indices]+=1
			# replace 'current' indices with 'least' indices:# if the removed mainindex is in 'least' or 'most' cathegory, replace its place with new value
			self.items_with_least_validscores_indices[np.where(self.eachweight[indices] > 0)[0]]= indices[self.eachweight[indices]>0]
			# remove mainindex if its weight gets zero
			indices2remove=indices[self.eachweight[indices]<=0]
			np.delete(self.eachweight, indices2remove)
			np.delete(self.nonzeroheadrelsinds, indices2remove)
		elif new_validation_score > self.items_with_most_validscores_score:
			# increase weights of 'most' and decrease 'current'
			self.eachweight[indices] += 1
			self.eachweight[self.items_with_least_validscores_indices] -= 1
			# replace 'current' indices with 'most' indices:# if the removed mainindex is in 'least' or 'most' cathegory, replace its place with new value
			self.items_with_most_validscores_indices[np.where(self.eachweight[indices] > 0)[0]] = indices[ self.eachweight[indices] > 0]
			# remove 'least' cath if its weight gets zero
			indices2remove = self.items_with_least_validscores_indices[self.eachweight[self.items_with_least_validscores_indices] <=0]
			np.delete(self.eachweight, indices2remove)
			np.delete(self.nonzeroheadrelsinds, indices2remove)
			self.items_with_least_validscores_indices[indices2remove]=np.floor(np.random.random((len(indices2remove)))*len(self.eachweight))
		elif new_validation_score <= self.items_with_most_validscores_score and new_validation_score >= self.items_with_least_validscores_score:
			# add to the 'most' subtract from 'current'
			self.eachweight[indices] -= 1
			self.eachweight[self.items_with_most_validscores_indices] += 1
			# remove mainindex if its weight gets zero
			indices2remove=indices[self.eachweight[indices]<=0]
			np.delete(self.eachweight, indices2remove)
			np.delete(self.nonzeroheadrelsinds, indices2remove)

	def __iter__(self):
		if self.mode=='train':
			import math
			import numpy as np
			# iterate over vector of indices inside iteration of maxiter , then yield
			self.indexindicator=0
			self.multipleratio=math.floor(len(self.nonzeroheadrelsinds[self.mode][0])/self.batchsize)*self.batchsize
			if self.samplertype == 'shuffle':
				while self.indexindicator<self.multipleratio:
					batch= np.concatenate([[self.nonzeroheadrelsinds[self.mode][0][self.indexindicator:self.indexindicator+self.batchsize]], [self.nonzeroheadrelsinds[self.mode][1][self.indexindicator:self.indexindicator+self.batchsize]] ],0 ).transpose()
					self.indexindicator+=  self.batchsize
					yield batch
				self.indexindicator=0
				batch= np.concatenate([[self.nonzeroheadrelsinds[self.mode][0][self.indexindicator:self.indexindicator+self.batchsize]], [self.nonzeroheadrelsinds[self.mode][1][self.indexindicator:self.indexindicator+self.batchsize]] ],0 ).transpose()
				self.indexindicator+=  self.batchsize
				return batch
			#
			elif self.samplertype== 'weighted':
				setlen=len(self.nonzeroheadrelsinds[self.mode][0])
				eachweightsampled=np.floor(np.random.random((self.batchsize))*setlen)
				while setlen>0:
					indicessampled_head, indicessampled_rel= self.nonzeroheadrelsinds[self.mode][0][eachweightsampled], self.nonzeroheadrelsinds[self.mode][1][eachweightsampled]
					batch= np.concatenate([[indicessampled_head], [indicessampled_rel] ],0 ).transpose()
					# update self.eachweight
					weight= self.eachweight[indicessampled_head, indicessampled_rel]-1
					indices2delete= eachweightsampled[np.where(weight<=0)[0]] # ifd err, makeit right
					self.eachweight[indicessampled_head, indicessampled_rel]-=1
					self.nonzeroheadrelsinds[self.mode][0], self.nonzeroheadrelsinds[self.mode][1] =np.delete(self.nonzeroheadrelsinds[self.mode][0],indices2delete) , np.delete(self.nonzeroheadrelsinds[self.mode][1],indices2delete)
					#
					yield batch
				# else:
				# again do everything from start!!!
				self.eachweight = np.floor(self.sumovertail*self.maxiter/(np.sum(self.sumovertail)/np.sum(self.sumovertail>0)))
				self.nonzeroheadrelsinds[self.mode]= self.sumovertail.nonzero()
				# return self.__iter__(self)
			#
			elif self.samplertype=='adaptive':
				setlen=len(self.nonzeroheadrelsinds[self.mode][0])
				eachweightsampled=np.floor(np.random.random((self.batchsize))*setlen)
				while setlen>0:
					indicessampled_head, indicessampled_rel= self.nonzeroheadrelsinds[self.mode][0][eachweightsampled], self.nonzeroheadrelsinds[self.mode][1][eachweightsampled]
					self.currentbatch= np.concatenate([[indicessampled_head], [indicessampled_rel] ],0 ).transpose()
					# update self.eachweight
					weight= self.eachweight[indicessampled_head, indicessampled_rel]-1
					indices2delete= eachweightsampled[np.where(weight<=0)[0]] # ifd err, makeit right
					self.eachweight[indicessampled_head, indicessampled_rel]-=1
					self.nonzeroheadrelsinds[self.mode][0], self.nonzeroheadrelsinds[self.mode][1] =np.delete(self.nonzeroheadrelsinds[self.mode][0],indices2delete) , np.delete(self.nonzeroheadrelsinds[self.mode][1],indices2delete)
					#
					yield self.currentbatch
				# # sample current batch randomly
				# # decrease weight of current batch
				# # check and remove those reached zero
				# setlen=len(self.eachweight)
				# self.currentlysampledindices=np.floor(np.random.random((self.batchsize))*setlen)
				# if setlen>0:
				#	 indicessampled_head, indicessampled_rel= self.nonzeroheadrelsinds[0][self.currentlysampledindices], self.nonzeroheadrelsinds[1][self.currentlysampledindices]
				#	 self.currentbatch= np.concatenate([[indicessampled_head], [indicessampled_rel] ],0 ).transpose()
				#	 # update self.eachweight
				#	 eachweightvalue= self.eachweight[self.currentlysampledindices]-1
				#	 indices2delete= self.currentlysampledindices[np.where(eachweightvalue<=0)[0]] # ifd err, makeit right
				#	 self.eachweight[self.currentlysampledindices]-=1
				#	 self.eachweight, self.nonzeroheadrelsinds[0], self.nonzeroheadrelsinds[1] = np.delete(self.eachweight, indices2delete), np.delete(self.nonzeroheadrelsinds[0],indices2delete) , np.delete(self.nonzeroheadrelsinds[1],indices2delete)
				#	 #
				#	 return self.currentbatch
				# else:
				# again do everything from start!!!
				self.nonzeroheadrelsinds[self.mode] = self.sumovertail.nonzero()
				self.eachweight = np.ones(self.sumovertail.shape) * self.maxiter
				# return self.__iter__(self)
		elif self.mode=='valid':
			import math
			import numpy as np
			# iterate over vector of indices inside iteration of maxiter , then yield
			self.Valid_indexindicator = 0
			self.Valid_multipleratio = math.floor(
				len(self.nonzeroheadrelsinds[self.mode][0]) / self.Valid_batchsize) * self.Valid_batchsize
			if self.samplertype == 'shuffle':
				while self.Valid_indexindicator < self.Valid_multipleratio:
					batch = np.concatenate([[self.nonzeroheadrelsinds[self.mode][0][
											 self.Valid_indexindicator:self.Valid_indexindicator + self.Valid_batchsize]],
											[self.nonzeroheadrelsinds[self.mode][1][
											 self.Valid_indexindicator:self.Valid_indexindicator + self.Valid_batchsize]]],
										   0).transpose()
					self.Valid_indexindicator += self.Valid_batchsize
					yield batch
				self.Valid_indexindicator = 0
				batch = np.concatenate([[self.nonzeroheadrelsinds[self.mode][0][
										 self.Valid_indexindicator:self.Valid_indexindicator + self.Valid_batchsize]], [
											self.nonzeroheadrelsinds[self.mode][1][
											self.Valid_indexindicator:self.Valid_indexindicator + self.Valid_batchsize]]],
									   0).transpose()
				self.Valid_indexindicator += self.Valid_batchsize
				return batch
			#
			elif self.samplertype == 'weighted':
				setlen = len(self.nonzeroheadrelsinds[self.mode][0])
				eachweightsampled = np.floor(np.random.random(self.Valid_batchsize) * setlen)
				while setlen > 0:
					indicessampled_head, indicessampled_rel = self.nonzeroheadrelsinds[self.mode][0][eachweightsampled], \
															  self.nonzeroheadrelsinds[self.mode][1][eachweightsampled]
					batch = np.concatenate([[indicessampled_head], [indicessampled_rel]], 0).transpose()
					# update self.eachweight
					weight = self.eachweight[indicessampled_head, indicessampled_rel] - 1
					indices2delete = eachweightsampled[np.where(weight <= 0)[0]]  # ifd err, makeit right
					self.eachweight[indicessampled_head, indicessampled_rel] -= 1
					self.nonzeroheadrelsinds[self.mode][0], self.nonzeroheadrelsinds[self.mode][1] = np.delete(
						self.nonzeroheadrelsinds[self.mode][0], indices2delete), np.delete(
						self.nonzeroheadrelsinds[self.mode][1], indices2delete)
					#
					yield batch
				# else:
				# again do everything from start!!!
				self.eachweight = np.floor(
					self.Valid_sumovertail * self.maxiter / (np.sum(self.sumovertail) / np.sum(self.sumovertail > 0)))
				self.nonzeroheadrelsinds[self.mode] = self.sumovertail.nonzero()
				# return self.__iter__(self)
			#
			elif self.samplertype == 'adaptive':
				setlen = len(self.nonzeroheadrelsinds[self.mode][0])
				eachweightsampled = np.floor(np.random.random((self.Valid_batchsize)) * setlen)
				while setlen > 0:
					indicessampled_head, indicessampled_rel = self.nonzeroheadrelsinds[self.mode][0][eachweightsampled], \
															  self.nonzeroheadrelsinds[self.mode][1][eachweightsampled]
					self.Valid_currentbatch = np.concatenate([[indicessampled_head], [indicessampled_rel]],
															 0).transpose()
					# update self.eachweight
					weight = self.eachweight[indicessampled_head, indicessampled_rel] - 1
					indices2delete = eachweightsampled[np.where(weight <= 0)[0]]  # ifd err, makeit right
					self.eachweight[indicessampled_head, indicessampled_rel] -= 1
					self.nonzeroheadrelsinds[self.mode][0], self.nonzeroheadrelsinds[self.mode][1] = np.delete(
						self.nonzeroheadrelsinds[self.mode][0], indices2delete), np.delete(
						self.nonzeroheadrelsinds[self.mode][1], indices2delete)
					#
					yield self.Valid_currentbatch
				# again do everything from start!!!
				self.nonzeroheadrelsinds[self.mode] = self.sumovertail.nonzero()
				self.eachweight = np.ones(self.sumovertail.shape) * self.maxiter
				# return self.__iter__(self)
		elif self.mode=='test':
			import math
			import numpy as np
			# iterate over vector of indices inside iteration of maxiter , then yield
			self.Test_indexindicator = 0
			self.Test_multipleratio = math.floor(
				len(self.nonzeroheadrelsinds[self.mode][0]) / self.Test_batchsize) * self.Test_batchsize
			if self.samplertype == 'shuffle':
				while self.Test_indexindicator < self.Test_multipleratio:
					batch = np.concatenate([[self.nonzeroheadrelsinds[self.mode][0][
											 self.Test_indexindicator:self.Test_indexindicator + self.Test_batchsize]],
											[self.nonzeroheadrelsinds[self.mode][1][
											 self.Test_indexindicator:self.Test_indexindicator + self.Test_batchsize]]],
										   0).transpose()
					self.Test_indexindicator += self.Test_batchsize
					yield batch
				self.Test_indexindicator = 0
				batch = np.concatenate([[self.nonzeroheadrelsinds[self.mode][0][
										 self.Test_indexindicator:self.Test_indexindicator + self.Test_batchsize]],
										[self.nonzeroheadrelsinds[self.mode][1][
										 self.Test_indexindicator:self.Test_indexindicator + self.Test_batchsize]]],
									   0).transpose()
				self.Test_indexindicator += self.Test_batchsize
				return batch
			#
			elif self.samplertype == 'weighted':
				setlen = len(self.nonzeroheadrelsinds[self.mode][0])
				eachweightsampled = np.floor(np.random.random(self.Test_batchsize) * setlen)
				while setlen > 0:
					indicessampled_head, indicessampled_rel = self.nonzeroheadrelsinds[self.mode][0][
																  eachweightsampled], \
															  self.nonzeroheadrelsinds[self.mode][1][
																  eachweightsampled]
					batch = np.concatenate([[indicessampled_head], [indicessampled_rel]], 0).transpose()
					# update self.eachweight
					weight = self.eachweight[indicessampled_head, indicessampled_rel] - 1
					indices2delete = eachweightsampled[np.where(weight <= 0)[0]]  # ifd err, makeit right
					self.eachweight[indicessampled_head, indicessampled_rel] -= 1
					self.nonzeroheadrelsinds[self.mode][0], self.nonzeroheadrelsinds[self.mode][1] = np.delete(
						self.nonzeroheadrelsinds[self.mode][0], indices2delete), np.delete(
						self.nonzeroheadrelsinds[self.mode][1], indices2delete)
					#
					yield batch
				# else:
				# again do everything from start!!!
				self.eachweight = np.floor(self.Test_sumovertail * self.maxiter / (
							np.sum(self.sumovertail) / np.sum(self.sumovertail > 0)))
				self.nonzeroheadrelsinds[self.mode] = self.sumovertail.nonzero()
				# return self.__iter__(self)
			#
			elif self.samplertype == 'adaptive':
				setlen = len(self.nonzeroheadrelsinds[self.mode][0])
				eachweightsampled = np.floor(np.random.random((self.Test_batchsize)) * setlen)
				while setlen > 0:
					indicessampled_head, indicessampled_rel = self.nonzeroheadrelsinds[self.mode][0][
																  eachweightsampled], \
															  self.nonzeroheadrelsinds[self.mode][1][
																  eachweightsampled]
					self.Test_currentbatch = np.concatenate([[indicessampled_head], [indicessampled_rel]],
															0).transpose()
					# update self.eachweight
					weight = self.eachweight[indicessampled_head, indicessampled_rel] - 1
					indices2delete = eachweightsampled[np.where(weight <= 0)[0]]  # ifd err, makeit right
					self.eachweight[indicessampled_head, indicessampled_rel] -= 1
					self.nonzeroheadrelsinds[self.mode][0], self.nonzeroheadrelsinds[self.mode][1] = np.delete(
						self.nonzeroheadrelsinds[self.mode][0], indices2delete), np.delete(
						self.nonzeroheadrelsinds[self.mode][1], indices2delete)
					#
					yield self.Test_currentbatch
				# again do everything from start!!!
				self.nonzeroheadrelsinds[self.mode] = self.sumovertail.nonzero()
				self.eachweight = np.ones(self.sumovertail.shape) * self.maxiter
				# return self.__iter__(self)









def readLines(filepath,_from=None, _to=None, _list=None):
	'''
	read from text by list with specified  line indices
	:param filepath: Path to file
	:param _list: sorted list of line numbers
	:returns strpool: list of strings associated to indices in _list
	'''
	strpool=[]
	if _list==None:
		if _from==None:
			_list=list(range(0,1))
		else:
			if _to!=None:
				_list=list(range(_from,_to))
			else:
				_list=[_from]
	elif type(_list)!=type(list()):
		_list= list(_list)
	#
	pointer=0
	with open(filepath) as fp:
		for i, line in enumerate(fp):
			if i==_list[pointer]:
				strpool.append(line)
				pointer+=1
	return strpool



class arg():
	pass

def prompt4args(args=arg()):
	import torch
	##TODO: import tuner here and define its ingredients
	import pickle
	from dim_reductions import load_dimreduction
	from makeDataReady2 import selectBestClusterIndices4EntitiessOverBenchmark0_tailke20tabegirim , selectBestClusterIndices4RelationsOverBenchmark
	from datastructs import tensor3d
	# args=arg()
	args.db='freebase_proced_'
	#todo: tune it
	args.datatype ='weighted' # ['weighted','unnestedrels','unweightednestsinglecount']
	args.path='../data/'#'/home/mm/Documents/myprojs/2_projB_2/'
	# load entity data:
	with open(args.path+args.db+args.datatype+'_entt_1500features.pkl','rb') as fff:
		args.entities=pickle.load(fff)
	#
	with open(args.path+args.db+args.datatype+'#relations75x20features2.pkl','rb') as fff:
		args.relations=pickle.load(fff)
	# extract dimentionality reduction transform for chosen tune
	args.dimredtype_entities='pca' # #todo tune it: 'svd','missingvalueratio','fastica','factoranal','isomap','lowvarfilter','spectralembed'
	args.dimredtype_relations='pca' # #todo tune it: 'svd','missingvalueratio','fastica','factoranal','isomap','lowvarfilter','spectralembed'
	args.entities_dimentionality_reduction=  load_dimreduction(name=args.dimredtype_entities, type='entities',database=args.db,path=args.path)
	args.relations_dimentionality_reduction= load_dimreduction(name=args.dimredtype_relations,type='relations',database=args.db,path=args.path)
	# clusters of entities relations:
	args.entities_clusters_id= selectBestClusterIndices4EntitiessOverBenchmark0_tailke20tabegirim(args.db, args.datatype,args.path)
	args.relations_clusters_id= selectBestClusterIndices4RelationsOverBenchmark(args.db, args.datatype,args.path)
	# load nameoftensor of headreltail glossaries (with rel as default):
	args.tails_glossary_path = args.path + args.db + 'pickle_tnsr_4clusteringrelations_4embeddingreductionofentities_'+args.datatype+'.pkl'
	args.tails_glossary_tensor=tensor3d().load(args.tails_glossary_path)
	###
	# <to_tune> :
	#$#
	if 'negative_sampler_thresh' not in vars(args):
		args.negative_sampler_thresh= 0.1# 0.6 0.8 0.9
	if 'projElossfcnType' not in vars(args):
		args.projElossfcnType='listwise_sumoverbatch'#'listwise'  'pointwise'
	args.cluster_update_regularity= 4 # 2 4 6 8 10
	args.pretrained_entities_dimentionality_reduction= False # false:i want model to train # True
	args.pretrained_relations_dimentionality_reduction= False # false:i want model to train # True
	args.normalize_candidate_tails= True
	args.normalize_candidate_heads= False
	args.regularize_within_clusters_relations= False
	args.regularize_within_clusters_entities= False
	args.regularization_rate_relations = 0.0001  ##TODO:tune
	args.regularization_rate_entities  = 0.0001 ##TODO:tune
	args.reldim=100
	args.entdim=100
	###  Train validation test segments:
	args.trainsegment=0.7
	args.validsegment=0.2
	args.testsegment=0.1
	# </to_tune>
	# optimization args:
	args.batchsize=20 #todo: Felan 1 .# todo:TUNE
	args.testbatchsize=100
	args.validbatchsize=100
	if 'learningrate' not in vars(args):
		args.learningrate=0.01# todo:TUNE ##TODO: if nothing happened this time, just remove sumoverheads tails
	args.momentum = 0.7 # todo:TUNE
	args.weight_decay = 1e-5# todo:TUNE
	args.batchnorm = False  #todo: not implemented yet
	if 'batchsampler' not in vars(args):
		args.batchsampler = 'shuffle' #weighted #adaptive #todo:not debugged yet :  shuffle/attention / nearestneighbor / ada (adaptive reweighting(increase/decrease) sampler prob based on validation err)
	args.earlystopimprovethresh=0.01 # todo:not implemented yet
	args.maxiter=3 #
	args.testmaxiter = 1
	args.validmaxiter = 1
	args.mode='train'
	if torch.cuda.is_available():
		args.cudaenabled =True#False
	else:
		args.cudaenabled = False
	return args




class tuner():
	def __init__(self,*tunes):
		names=[]
		# get names+  for each tune, index name of key:
		tunes_index=[[] for _ in tunes]
		for ii,tune in enumerate(tunes):
			keys=list(tune.keys())
			for jj,key in enumerate(keys):
				if key not in names:
					names.append(key)
				tunes_index[ii].append(names.index(key))
		#
		tuneslist=[[[]] for _ in tunes]
		keyslist=[[[]]  for _ in tunes]
		newtuneslist=[[] for _ in tunes]
		newkeyslist=[[]  for _ in tunes]
		currentcnt=1
		# get list:
		for i,(tune,tuneindex) in enumerate(zip(tunes,tunes_index)):
			remaineddict=tune.copy()
			for indofnameind, nameind in enumerate( tuneindex):  # key,values in zip(remaineddict.keys(), remaineddict.values()):
				key = names[nameind]
				values = tune[key]
				for value in values:
					for keyy,itm in zip(keyslist[i],tuneslist[i]):
						newtuneslist[i].append([*itm, value ])
						newkeyslist[i].append([*keyy, key ])
				tuneslist[i]= newtuneslist[i].copy()
				newtuneslist[i]=[]
				keyslist[i]= newkeyslist[i].copy()
				newkeyslist[i]=[]

		# concat results :
		tuness = []
		for itm in tuneslist:
			tuness.extend(itm)
		keyss=[]
		for itm in keyslist:
			keyss.extend(itm)

		# return list of dicts:
		self.listoftunes=[dict(zip(itmkeys,itmvalues)) for itmkeys,itmvalues in zip(keyss,tuness)]
		self.currentindex=-1



	def __iter__(self):
		for self.currentindex in range(self.currentindex,len(self.listoftunes)):
			self.tune= self.listoftunes[self.currentindex]
			yield self.applytune()


	def applytune(self):
		args = arg()
		for itmname,itmvalue in zip(self.tune.keys(),self.tune.values()):
			exec('args.'+itmname+'='+itmvalue)
		return args




def main_process_wo_tuner():
	from model import model_b
	from model import model_b_alldataingpu_experimental_forlowmemorygpu,model_b_alldataingpu_experimental
	import pickle
	import shutil
	import torch
	import time
	debugyet=0
	args= prompt4args() # func where u're free2 define all dataset infos and what to out
	nextbatchiter=batchloader(args) # batchloader class that each time provides next iter of features as tensor
	# load model:
	if debugyet ==0:
		with open('/data/bench_tmpmodeldata.pkl', 'rb') as ws:
			torchmodel=pickle.load(ws)
			# torchmodel.negative_sampler_thresh=0.0005

	else:
		torchmodel=model_b_alldataingpu_experimental(args)		#model_b_alldataingpu_experimental_forlowmemorygpu(args) #model_b_alldataingpu_experimental(args)		# torchmodel=model_b(args)
		## Just save it in a pickle, not to recompute things again:
		with open('../data/bench_tmpmodeldata.pkl','wb') as ws:
			pickle.dump( torchmodel  ,ws)
	keeptrackofbatches=[]
	keeptrackoflosses=[]
	keeptrackofbatches_valid=[]
	keeptrackoflosses_valid=[]
	totelapsedtimes=[]
	start_time=time.time()
	for it in range(nextbatchiter.maxiter):
		torchmodel.setmode('train')
		nextbatchiter.setmode('train')
		for i,batch in enumerate(nextbatchiter):
			# TODO: check if necesary to use model.train() as switching to train mode or model.eval() for switching to eval mode
			# train batch on model & get loss for each subbatch
			loss_foreachsubbatch= torchmodel.forward_batch(batch)#torchmodel.forward(batch)
			keeptrackofbatches.append(batch)
			keeptrackoflosses.append(loss_foreachsubbatch)
			# show train scores
			# get validation scores + for case of 'adaptive' sampling
			if i%100==99:
				elpsed=time.time()-start_time
				totelapsedtimes.append(elpsed)
				start_time = time.time()
				print('currently: %3dth epoch ||| %3dth iter ||| %3dth data.. num of args are %3d. ||| Elapsedtime:%3f'%(i,nextbatchiter.multipleratio,it, nextbatchiter.indexindicator,elpsed ))
				if i%500==499:
					shutil.copyfile('../data/bench_tmpmodeldata2.pkl','../data/bench_tmpmodeldata2_lastbck.pkl')
					with open('../data/bench_tmpmodeldata2.pkl', 'wb') as ws:
						pickle.dump([torchmodel.E , torchmodel.R , torchmodel.CE , torchmodel.CR,torchmodel.dimred_r,torchmodel.dimred_e ,totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses],ws)# pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)
					# with open('tmpmodeldata2_oth.pkl', 'wb') as ws:
					#	 pickle.dump([args,nextbatchiter], ws)  # pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)
		#
		# get validation score:
		torchmodel.setmode('valid')
		nextbatchiter.setmode('valid')
		for u,btch in enumerate(nextbatchiter):
			# if u>40:
			#	 break
			loss_foreachsubbatch_valid = torchmodel.forward(btch)
			keeptrackofbatches_valid.append(btch)
			keeptrackoflosses_valid.append(loss_foreachsubbatch_valid)

			if args.batchsampler=='adaptive':
				pass
				# reweight samples in adaptive sampler
				#### (Not implemented yet) nextbatchiter.update_eachweight(loss_foreachsubbatch, new_validation_score )

			# update clusters based on resulted embeds every model.cluster_update_regularity time by once ##TODO
			#### (Not implemented yet)

			# import pickle
			# with open('/home/mm/Documents/myprojs/2_projB_2/tmpmodeldata2.pkl','rb') as aa:
			#	 [E, R, CE, CR,   totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses]=pickle.load(aa)
			#




def main_process_tuner():
	from model import model_b
	from model import model_b_alldataingpu_experimental_forlowmemorygpu,model_b_alldataingpu_experimental
	import pickle
	import shutil
	import torch
	import numpy as np
	import time
	from utils import generateTunes
	resumeFrombreakpoint=1 # SET CHECKED IF PREMATURE STOP HAPPENED
	###
	# Tuner geneneration Phase:
	# tunes={ 'batchsampler':['"'"shuffle"'"'],'learningrate':['0.05','0.1','0.5'],'negative_sampler_thresh':['0.003','0.001','0.009'] , 'projElossfcnType':['"'"listwise_sumoverbatch"'"', '"'"pointwise"'"'] }
	tunes={ 'batchsampler':['"'"shuffle"'"'],'negative_sampler_thresh':['0.1','0.3','0.6','0.01','0.004'] , 'projElossfcnType':['"'"listwise_sumoverbatch"'"', '"'"pointwise"'"'], 'normalize_candidate_heads':['False','True'], 'regularize_within_clusters_relations':['False','True'], 'regularize_within_clusters_entities':['False','True'] }
	tuner_results=[]
	iternum=[]
	inum=[]
	teststate=[]
	tunerstate=[]
	sumloss=[]
	validation_info={}
	training_info={}
	all_tunes_results= []
	notnessessarythistime = False
	profiler=''
	tunerobj=tuner(tunes)
	if resumeFrombreakpoint==1:
		try:
			with open('../data/bench_tmpmodeldata2.pkl', 'rb') as w:
				# tuner_results,currentindex, tunerstate, inum, iternum = pickle.load(w)
				tuner_results,currentindex, _, inum, iternum ,training_info,all_tunes_results,profiler= pickle.load(w)
				# notnessessarythistime=True
				#
				score_listwise , score_pointwise , detectionrate_listwise , detectionrate_pointwise , posloss_listwise , negloss_listwise , regulloss_listwise , posloss_pointwise , negloss_pointwise , regulloss_pointwise , sum_listwise, sum_pointwise, _= training_info.values()
				score_pointwise*=sum_pointwise
				score_listwise*=sum_listwise
				detectionrate_listwise*=sum_listwise
				detectionrate_pointwise*=sum_pointwise
				posloss_listwise*=sum_listwise
				negloss_listwise*=sum_listwise
				regulloss_listwise*=sum_listwise
				posloss_pointwise*=sum_pointwise
				negloss_pointwise*=sum_pointwise
				regulloss_pointwise*=sum_pointwise
				#
				inum=[inum[-1]]
				iternum=[iternum[-1]]
				# tuner_results=[tuner_results[-1]]#[tuner_results[-1]] pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)
				tuner_results=[tuner_results]#[tuner_results[-1]] pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)
				tunerobj.currentindex=currentindex
		except:
			resumeFrombreakpoint=0
			print('WARNING.... NO SAVED DATA FOUND!!!')
			print('WARNING.... NO SAVED DATA FOUND!!!')
			print('WARNING.... NO SAVED DATA FOUND!!!')	
		
	for args_of_tune in tunerobj:
		args= prompt4args(args_of_tune) # func where u're free2 define all dataset infos and what to out
		args.regularize_within_clusters_relations= False
		args.regularize_within_clusters_entities= False
		args.batchsize=80 ##TODO
		
		nextbatchiter=batchloader(args) # batchloader class that each time provides next iter of features as tensor
		# load model:
		torchmodel = model_b_alldataingpu_experimental(args)
		tunerstate.append(args_of_tune)
		keeptrackofbatches=[]
		keeptrackoflosses=[]
		keeptrackofbatches_valid=[]
		keeptrackoflosses_valid=[]
		keeptrackofdetectionscore_valid=[]
		loss_foreachsubbatch=[]
		totelapsedtimes=[]
		if resumeFrombreakpoint == 1:
			if currentindex > tunerobj.currentindex:
				continue
			elif currentindex == tunerobj.currentindex:
				torchmodel.negative_sampler_thresh= args.negative_sampler_thresh #0.1 0.6 0.8 0.9
				torchmodel.normalize_candidate_tails= args.normalize_candidate_tails
				torchmodel.normalize_candidate_heads= args.normalize_candidate_heads#False
				torchmodel.regularize_within_clusters_relations= args.regularize_within_clusters_relations# False
				torchmodel.regularize_within_clusters_entities= args.regularize_within_clusters_entities#False
				torchmodel.load_tuner_results(tuner_results)
				print('loaded_modelparams')
		start_time=time.time()
		for it in range(nextbatchiter.maxiter):
			batchstarttime=time.time()
			if resumeFrombreakpoint == 1:
				if it<iternum[-1] and currentindex==tunerobj.currentindex:
					continue
			torchmodel.setmode('train')
			nextbatchiter.setmode('train')
			detectionrate_pointwise,score_listwise,score_pointwise,detectionrate_listwise, posloss_listwise, negloss_listwise, regulloss_listwise,posloss_pointwise, negloss_pointwise, regulloss_pointwise,sum_listwise,sum_pointwise=0,0,0,0,0,0,0,0,0,0,0,0
			for i,batch in enumerate(nextbatchiter):
				if resumeFrombreakpoint == 1:
					if it == iternum[-1] and currentindex == tunerobj.currentindex and i<inum[-1]:
						continue
				lossnegnp_profile, lossposnp, lossregulsnp, hasdetected, detectionscore, numoftailsinthisscore = torchmodel.forward_batch(batch)
				keeptrackofbatches.extend(batch)
				sum_listwise=i+1
				sum_pointwise+= numoftailsinthisscore
				score_listwise+=detectionscore
				score_pointwise+=detectionscore*numoftailsinthisscore
				detectionrate_listwise+=hasdetected
				detectionrate_pointwise+=hasdetected*numoftailsinthisscore
				posloss_listwise+=lossposnp
				posloss_pointwise+=lossposnp*numoftailsinthisscore
				negloss_listwise+=lossnegnp_profile
				negloss_pointwise+=lossnegnp_profile*numoftailsinthisscore
				regulloss_listwise+=lossregulsnp
				regulloss_pointwise+=lossregulsnp*numoftailsinthisscore

				if i%500==499:
					training_info=dict(zip( ['TRAINscore_listwise','TRAINscore_pointwise','TRAINdetectionrate_listwise','TRAINdetectionrate_pointwise','TRAINposloss_listwise','TRAINnegloss_listwise','TRAINregulloss_listwise','TRAINposloss_pointwise','TRAINnegloss_pointwise','TRAINregulloss_pointwise','TRAINsum_listwise','TRAINsum_pointwise__or_index','TRAINnumofvalidationinstances'] , [score_listwise/sum_listwise,score_pointwise/sum_pointwise,detectionrate_listwise/sum_listwise,detectionrate_pointwise/sum_pointwise, posloss_listwise/sum_listwise, negloss_listwise/sum_listwise, regulloss_listwise/sum_listwise,posloss_pointwise/sum_pointwise, negloss_pointwise/sum_pointwise, regulloss_pointwise/sum_pointwise,sum_listwise,sum_pointwise,nextbatchiter.multipleratio] ))
					elpsed=time.time()-start_time
					totelapsedtimes.append(elpsed)
					inum.append(i)
					iternum.append(it)
					start_time = time.time()
					what2print='>>>>>>>> currently: %3dth epoch ||| %3dth iter ||| %3dth data.. num of args are %3d. ||| Elapsedtime:%3f'%(i,nextbatchiter.multipleratio, nextbatchiter.indexindicator,it,elpsed )
					print(what2print)
					print(training_info)
					profiler=profiler+what2print+'\r\n\r\n'+str(training_info)+'\r\n\r\n\r\n'
				if i%2500==2499:
					try:
						shutil.copyfile('../data/bench_tmpmodeldata2.pkl','tmpmodeldata2_lastbck.pkl')
					except:
						pass
					# tuner_results.append([torchmodel.E , torchmodel.R , torchmodel.CE , torchmodel.CR,torchmodel.dimred_r,torchmodel.dimred_e ,totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses,sumloss])
					tuner_results=[torchmodel.E , torchmodel.R , torchmodel.CE , torchmodel.CR,torchmodel.dimred_r,torchmodel.dimred_e ,totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses,sumloss]
					all_tunes_results.append((dict(zip(['totelapsedtimes', 'loss_foreachsubbatch', 'keeptrackofbatches', 'keeptrackoflosses','sumloss','args_of_tune', 'it','i', 'tunerobjcurrentindex', 'tunerstate','inum','iternum','training_info'],[totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses,sumloss,args_of_tune, it,i, tunerobj.currentindex, tunerstate,inum,iternum,training_info])),'train'))
					with open('../data/bench_tmpmodeldata2.pkl', 'wb') as ws:
						pickle.dump([tuner_results,tunerobj.currentindex, tunerstate,inum,iternum,training_info,all_tunes_results,profiler],ws)# pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)
					with open('../data/bench_onlyreports.pkl', 'wb') as ws:
						pickle.dump([all_tunes_results,profiler],ws)# pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)
					
				if i % 5000 == 4999:
					backup_pydrive()
					print('backedup_again@')
			# AGAIN SAVE
			training_info=dict(zip( ['score_listwise','score_pointwise','detectionrate_listwise','detectionrate_pointwise','posloss_listwise','negloss_listwise','regulloss_listwise','posloss_pointwise','negloss_pointwise','regulloss_pointwise','sum_listwise','sum_pointwise__or_index','numofvalidationinstances'] , [score_listwise/sum_listwise,score_pointwise/sum_pointwise,detectionrate_listwise/sum_listwise,detectionrate_pointwise/sum_pointwise, posloss_listwise/sum_listwise, negloss_listwise/sum_listwise, regulloss_listwise/sum_listwise,posloss_pointwise/sum_pointwise, negloss_pointwise/sum_pointwise, regulloss_pointwise/sum_pointwise,sum_listwise,sum_pointwise,nextbatchiter.multipleratio] ))
			trainelapsedtime=time.time()-batchstarttime
			totelapsedtimes.append(trainelapsedtime)
			inum.append(i)
			iternum.append(it)
			start_time = time.time()
			print('currently: %3dth epoch ||| %3dth iter ||| %3dth data.. num of args are %3d. ||| Elapsedtime:%3f'%(i,nextbatchiter.multipleratio, nextbatchiter.indexindicator,it,elpsed ))
			print(training_info)
			try:
				shutil.copyfile('../data/bench_tmpmodeldata2.pkl','../data/bench_tmpmodeldata2_lastbck.pkl')
			except:# (ValueError,IOError) as err:
				pass
			# tuner_results.append([torchmodel.E , torchmodel.R , torchmodel.CE , torchmodel.CR,torchmodel.dimred_r,torchmodel.dimred_e ,totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses,sumloss])
			tuner_results=[torchmodel.E , torchmodel.R , torchmodel.CE , torchmodel.CR,torchmodel.dimred_r,torchmodel.dimred_e ,totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses,sumloss]
			all_tunes_results.append((dict(zip(['trainelapsedtime', 'loss_foreachsubbatch', 'keeptrackofbatches', 'keeptrackoflosses','sumloss','args_of_tune', 'it','i', 'tunerobjcurrentindex', 'tunerstate','inum','iternum','training_info'],[trainelapsedtime, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses,sumloss,args_of_tune, it,i, tunerobj.currentindex, tunerstate,inum,iternum,training_info])),'train'))
			with open('../data/bench_tmpmodeldata2.pkl', 'wb') as ws:
				pickle.dump([tuner_results,tunerobj.currentindex, tunerstate,inum,iternum,training_info,all_tunes_results,profiler],ws)# pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)
			with open('../data/bench_onlyreports.pkl', 'wb') as ws:
				pickle.dump([all_tunes_results,profiler],ws)# pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)
			#
			# get validation score:
			torchmodel.setmode('valid')
			nextbatchiter.setmode('valid')
			batchstarttime=time.time()
			detectionrate_pointwise,score_listwise,score_pointwise,detectionrate_listwise, posloss_listwise, negloss_listwise, regulloss_listwise,posloss_pointwise, negloss_pointwise, regulloss_pointwise,sum_listwise,sum_pointwise=0,0,0,0,0,0,0,0,0,0,0,0
			for u,btch in enumerate(nextbatchiter):
				lossnegnp, lossposnp, lossregulsnp, hasdetected, detectionscore, numoftailsinthisscore = torchmodel.forward_batch(btch)
				sum_listwise=u+1
				sum_pointwise+= numoftailsinthisscore
				score_listwise+=detectionscore
				score_pointwise+=detectionscore*numoftailsinthisscore
				detectionrate_listwise+=hasdetected
				detectionrate_pointwise+=hasdetected*numoftailsinthisscore
				posloss_listwise+=lossposnp
				posloss_pointwise+=lossposnp*numoftailsinthisscore
				negloss_listwise+=lossnegnp
				negloss_pointwise+=lossnegnp*numoftailsinthisscore
				regulloss_listwise+=lossregulsnp
				regulloss_pointwise+=lossregulsnp*numoftailsinthisscore
				if u%100==99:
					validation_info=dict(zip( ['TESTscore_listwise','TESTscore_pointwise','TESTdetectionrate_listwise','TESTdetectionrate_pointwise','TESTposloss_listwise','TESTnegloss_listwise','TESTregulloss_listwise','TESTposloss_pointwise','TESTnegloss_pointwise','TESTregulloss_pointwise','TESTsum_listwise','TESTsum_pointwise__or_index','TESTnumofvalidationinstances'] , [score_listwise/sum_listwise,score_pointwise/sum_pointwise,detectionrate_listwise/sum_listwise,detectionrate_pointwise/sum_pointwise, posloss_listwise/sum_listwise, negloss_listwise/sum_listwise, regulloss_listwise/sum_listwise,posloss_pointwise/sum_pointwise, negloss_pointwise/sum_pointwise, regulloss_pointwise/sum_pointwise,sum_listwise,sum_pointwise,nextbatchiter.Valid_multipleratio] ))
					profiler=profiler+'>>>ValidationStatusInTheMeanwhile:' + str(validation_info)+'\r\n\r\n'
					print(validation_info)
				#if u%5000==4999:
			validation_info=dict(zip( ['TESTscore_listwise','TESTscore_pointwise','TESTdetectionrate_listwise','TESTdetectionrate_pointwise','TESTposloss_listwise','TESTnegloss_listwise','TESTregulloss_listwise','TESTposloss_pointwise','TESTnegloss_pointwise','TESTregulloss_pointwise','TESTsum_listwise','TESTsum_pointwise__or_index','TESTnumofvalidationinstances'] , [score_listwise/sum_listwise,score_pointwise/sum_pointwise,detectionrate_listwise/sum_listwise,detectionrate_pointwise/sum_pointwise, posloss_listwise/sum_listwise, negloss_listwise/sum_listwise, regulloss_listwise/sum_listwise,posloss_pointwise/sum_pointwise, negloss_pointwise/sum_pointwise, regulloss_pointwise/sum_pointwise,sum_listwise,sum_pointwise,nextbatchiter.Valid_multipleratio] ))
			profiler=profiler+str(validation_info)+'\n\n'
			print(validation_info)
			testelapsedtime=time.time()-batchstarttime
			#with open('../data/bench_validres.pkl','wb') as ff:
			#	pickle.dump(validation_info,ff)
			#### saving validation data
			tuner_results=[torchmodel.E , torchmodel.R , torchmodel.CE , torchmodel.CR,torchmodel.dimred_r,torchmodel.dimred_e ,totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses,sumloss]
			all_tunes_results.append((dict(zip(['testelapsedtime', 'loss_foreachsubbatch', 'keeptrackofbatches', 'keeptrackoflosses','sumloss','args_of_tune', 'it','i', 'tunerobjcurrentindex', 'tunerstate','inum','iternum','training_info'],[testelapsedtime, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses,sumloss,args_of_tune, it,i, tunerobj.currentindex, tunerstate,inum,iternum,validation_info])),'valid'))
			with open('../data/bench_tmpmodeldata2.pkl', 'wb') as ws:
				pickle.dump([tuner_results,tunerobj.currentindex, tunerstate,inum,iternum,training_info,all_tunes_results,profiler],ws)# pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)
			with open('../data/bench_onlyreports.pkl', 'wb') as ws:
				pickle.dump([all_tunes_results,profiler],ws)# pickle.dump([torchmodel,i,args.indexindicator,args.indices], ws)

			torchmodel.setmode('train')
			nextbatchiter.setmode('train')





def testmodetest():
	from model import model_b
	from model import model_b_alldataingpu_experimental_forlowmemorygpu,model_b_alldataingpu_experimental
	import pickle
	import shutil
	import torch
	import numpy as np
	import time
	from utils import generateTunes
	import pickle
	debugyet=0
	args= prompt4args() # func where u're free2 define all dataset infos and what to out
	nextbatchiter=batchloader(args) # batchloader class that each time provides next iter of features as tensor
	# load model:
	if debugyet ==0:
		with open('../data/bench_tmpmodeldata.pkl', 'rb') as ws:
			torchmodel=pickle.load(ws)
			# torchmodel.negative_sampler_thresh=0.0005
	else:
		torchmodel=model_b_alldataingpu_experimental(args)		#model_b_alldataingpu_experimental_forlowmemorygpu(args) #model_b_alldataingpu_experimental(args)		# torchmodel=model_b(args)
		## Just save it in a pickle, not to recompute things again:
		with open('../data/bench_tmpmodeldata.pkl','wb') as ws:
			pickle.dump( torchmodel  ,ws)
	###
	# Tuner geneneration Phase:
	tunes={ 'batchsampler':['"'"shuffle"'"','"'"adaptive"'"'],'learningrate':['0.05','0.1','0.5'],'negative_sampler_thresh':['0.001','0.0006','0.0001'] , 'projElossfcnType':['"'"listwise_sumoverbatch"'"', '"'"pointwise"'"'] }
	tuner_results=[]
	iternum=[]
	inum=[]
	teststate=[]
	tunerstate=[]
	for args_of_tune in tuner(tunes):
		# args= prompt4args() # func where u're free2 define all dataset infos and what to out
		# nextbatchiter=batchloader(args) # batchloader class that each time provides next iter of features as tensor
		torchmodel.batchsize=20
		nextbatchiter.batchsize=20
		#
		tunerstate.append(args_of_tune)
		keeptrackofbatches=[]
		keeptrackoflosses=[]
		keeptrackofbatches_valid=[]
		keeptrackoflosses_valid=[]
		totelapsedtimes=[]
		start_time=time.time()
		for it in range(nextbatchiter.maxiter):
			torchmodel.setmode('valid')
			nextbatchiter.setmode('valid')
			for u,btch in enumerate(nextbatchiter):
				loss_foreachsubbatch_valid = torchmodel.forward_batch(btch)
				keeptrackofbatches_valid.append(btch)
				keeptrackoflosses_valid.append(loss_foreachsubbatch_valid)
				if u%4==3:
					elpsed = time.time() - start_time
					totelapsedtimes.append(elpsed)
					iternum.append(it)
					start_time = time.time()
					print(
						'currently: %3dth epoch ||| %3dth iter ||| %3dth data.. num of args are %3d. ||| Elapsedtime:%3f' % (
						u, nextbatchiter.multipleratio, nextbatchiter.indexindicator, it, elpsed))
					print(loss_foreachsubbatch_valid)
					print(torchmodel.average_batch_loss_vector)

			sumloss=np.sum(keeptrackoflosses_valid)
			if args.batchsampler=='adaptive':
				nextbatchiter.update_eachweight(torchmodel.average_batch_loss_vector, sumloss)
				pass
			torchmodel.setmode('train')
			nextbatchiter.setmode('train')
			# show test scores

		pass





if __name__=="__main__":
	# main_process_wo_tuner()
	from pydrive_stuff import *
	main_process_tuner()
	# testmodetest()