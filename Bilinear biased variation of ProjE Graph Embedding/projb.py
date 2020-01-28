import torch as t
import torch.nn as nn
import numpy as np
from utils import xavier_init
from datastructs import Vocab

class model_b_alldataingpu_experimental(t.nn.Module): #@# model with learnable embedding +clusterbasedParamterershift+ bilinear trans
    def __init__(self, args):
        super(model_b_alldataingpu_experimental,self).__init__()
        """
        Creating   model.
        """
        self.cudaenabled=args.cudaenabled
        # general parameters
        self.mode=args.mode
        self.trainbatchsize=args.batchsize # todo: Felan 1 .# todo:TUNE
        self.validbatchsize=args.validbatchsize # todo: Felan 1 .# todo:TUNE
        self.testbatchsize=args.testbatchsize # todo: Felan 1 .# todo:TUNE
        if self.mode=='train':
            self.batchsize=self.trainbatchsize
        elif self.mode=='valid':
            self.batchsize=self.validbatchsize
        else:
            self.batchsize=self.testbatchsize
        self.learningrate =args.learningrate # todo:TUNE
        self.momentum =args.momentum   # todo:TUNE
        self.weight_decay =args.weight_decay  # todo:TUNE
        self.batchnorm =args.batchnorm # todo: not implemented yet
        self.batchsampler =args.batchsampler   # weighted #adaptive #todo:not debugged yet :  shuffle/attention / nearestneighbor / ada (adaptive reweighting(increase/decrease) sampler prob based on validation err)
        self.earlystopimprovethresh =args.earlystopimprovethresh  # todo:not implemented yet
        self.maxiter = args.maxiter   #
        # glossaries to load
        self.datatype= args.datatype #['weighted','unnestedrels','unweightednestsinglecount'] #, 'unweightednest'
        self.entities_embeds=args.entities.todense()#self.fromnumpy(args.entities.todense()) #t.from_numpy(args.entities.todense())
        self.relations_embeds =args.relations.todense()#self.fromnumpy(args.relations.todense() )#t.from_numpy(args.relations.todense())
        self.edim= self.entities_embeds.shape[0]
        self.rdim= self.relations_embeds.shape[0]
        self.reldim=args.reldim
        self.entdim=args.entdim
        # todo: revise for better memory: tmp=vocab; del vocab
        self.entities_clusters_id= args.entities_clusters_id#
        self.id2clustersid4entities= Vocab(self.entities_clusters_id).word2id#args.entities_clusters_id
        self.relations_clusters_id= args.relations_clusters_id#
        self.id2clustersid4relations= Vocab(self.relations_clusters_id).word2id
        self.N_c_e= len(self.id2clustersid4entities)
        self.N_c_r= len(self.id2clustersid4relations)
        #
        self.tails_glossary_tensor= args.tails_glossary_tensor#self.fromsparse(args.tails_glossary_tensor.data ) #todo
        self.hlen, self.rlen, self.tlen= args.tails_glossary_tensor.hlen, args.tails_glossary_tensor.rlen, args.tails_glossary_tensor.tlen
        self.negative_sampler_thresh = args.negative_sampler_thresh ##TODO:tune
        self.projElossfcnType= args.projElossfcnType ##TODO:tune
        self.cluster_update_regularity = args.cluster_update_regularity ##TODO:tune
        self.pretrained_entities_dimentionality_reduction=args.pretrained_entities_dimentionality_reduction ##TODO:tune
        self.pretrained_relations_dimentionality_reduction=args.pretrained_relations_dimentionality_reduction ##TODO:tune
        self.normalize_candidate_tails=args.normalize_candidate_tails # True or false ##TODO:tune, reweight all tails DOWN if corresponding head(~input) are dense in numof tails
        self.normalize_candidate_heads=args.normalize_candidate_heads # True or false## ##TODO:tune, reweight each tail DOWN if its connected more densely to more heads
        self.sumoverheads= args.tails_glossary_tensor.sumover('heads')
        self.sumovertails= args.tails_glossary_tensor.sumover('tails')
        # self.sumoverheads= self.fromnumpy(args.tails_glossary_tensor.sumover('heads').todense())
        # self.sumovertails= self.fromnumpy(args.tails_glossary_tensor.sumover('tails').todense())
        self.regularize_within_clusters_relations= args.regularize_within_clusters_relations ##TODO:tune
        self.regularize_within_clusters_entities= args.regularize_within_clusters_entities ##TODO:tune
        self.regularization_rate_relations= args.regularization_rate_relations ##TODO:tune
        self.regularization_rate_entities= args.regularization_rate_entities ##TODO:tune
        # glossaries to learn
        self.E = self.fromnumpy(xavier_init((self.edim,self.entdim))) # entity vecs to indirectly learn
        self.R = self.fromnumpy(xavier_init((self.rdim,self.reldim))) # relation vecs to indirectly learn
        self.CR= self.fromnumpy(xavier_init((self.N_c_r, self.reldim))) # Parameters of all RELATIONS' clusters to indirectly learn
        self.CE= self.fromnumpy(xavier_init((self.N_c_e, self.entdim))) # Parameters of all ENTITIES' clusters to indirectly learn
        # Now set parameters : # t.nn.Parameter sets requires_grad true /also easily lists all gradablevars by model.parameters()
        self.e=self.tnnparameter(t.from_numpy(args.entities[0,:].todense())) #entity_current_input  defaultly,requires_grad=True
        self.r=self.tnnparameter(t.from_numpy(args.relations[0,:].todense())) # relation_current_input # defaultly,requires_grad=True
        self.cr=self.tnnparameter(self.CR[0,:].clone()) # entities_clusters # defaultly,requires_grad=True
        self.ce=self.tnnparameter(self.CE[0,:].clone()) # relations_clusters # defaultly,requires_grad=True
        # set parameters that don't need update_entities_relations_parameters
        self.bp=self.tnnparameter(t.from_numpy(xavier_init((1, self.entdim)) )) # projection bias # defaultly,requires_grad=True
        # dimensionality reduction, entities' feature  glossaries
        if self.pretrained_entities_dimentionality_reduction is True:
            if args.entities_dimentionality_reduction is not None:
                self.dimred_e = self.fromnumpy(args.entities_dimentionality_reduction)
            else:
                self.dimred_e = self.fromnumpy(xavier_init((args.entities.shape[1], self.edim)))
        else:
            if args.entities_dimentionality_reduction is not None:
                self.dimred_e = self.tnnparameter(t.from_numpy(args.entities_dimentionality_reduction))
            else:
                self.dimred_e = self.tnnparameter(t.from_numpy(xavier_init((args.entities.shape[1], self.edim))))
        # dimensionality reduction, relations' feature  glossaries
        if self.pretrained_relations_dimentionality_reduction is True:
            if args.relations_dimentionality_reduction is not None:
                self.dimred_r = self.fromnumpy(args.relations_dimentionality_reduction)
            else:
                self.dimred_r = self.fromnumpy(xavier_init((args.relations.shape[1], self.rdim)))
        else:
            if args.entities_dimentionality_reduction is not None:
                self.dimred_r = self.tnnparameter(t.from_numpy(args.relations_dimentionality_reduction))
            else:
                self.dimred_r = self.tnnparameter(t.from_numpy(xavier_init((args.relations.shape[1], self.rdim))))
        # current ongoing entity/rel/ParamRegardClus to learn
        self.entity_current_id= 0
        self.relation_current_id= 0
        self.entity_cluster_current_id= 0
        self.relation_cluster_current_id= 0
        # define loss function for model:
        if self.projElossfcnType=='pointwise':
            self.projBfcntype= t.sigmoid
        elif self.projElossfcnType== 'listwise' or self.projElossfcnType == 'listwise_sumoverbatch':
            self.projBfcntype= t.nn.functional.softmax
        #
        self.reset_averagebatchloss()

    def load_tuner_results(self,tuner_results):
        self.E, self.R, self.CE, self.CR, self.dimred_r, self.dimred_e,_, _, _, _, _ =tuner_results[-1]

    def reset_averagebatchloss(self):
        # some minimum and some maximum of losses during this run (halfiter)
        self.average_batch_loss_vector=[0 for _ in range(self.batchsize)]

    def fromsparse(self, mtx):
        import torch
        import numpy as np
        from scipy import sparse
        from scipy.sparse import coo_matrix
        coo=mtx.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        if self.cudaenabled == True:
            return torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()
        else:
            return torch.sparse.FloatTensor(i, v, torch.Size(shape))


    def fromnumpy(self,dat):
        import torch as t
        if self.cudaenabled==True:
            return t.from_numpy(dat).cuda()
        else:
            return t.from_numpy(dat)

    def tnnparameter(self,tens):
        import torch as t
        if self.cudaenabled==True:
            return t.nn.Parameter(tens.cuda())
        else:
            return t.nn.Parameter(tens)

    def loadcheckpoint(self,out=[]):
        import pickle
        with open('../data/bench_tmpmodeldata2.pkl', 'rb') as ws:
            [tuner_results, tunerstate, inum, iternum]=pickle.load(ws)
            self.E, self.R, self.CE, self.CR, self.dimred_r, self.dimred_e,totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses, sumloss=tuner_results
            out=[tuner_results,totelapsedtimes, loss_foreachsubbatch, keeptrackofbatches, keeptrackoflosses, sumloss]


    def tensors2cuda(self):
        for itm in self.__dict__.keys():
            if type(self.__dict__[itm]) is t.Tensor:
                self.__dict__[itm]= self.__dict__[itm].cuda()

    def setmode(self,mode='train'):
        if mode=='train' or mode=='training':
            self.mode='train'
        elif mode=='test' or mode=='testing' or mode=='valid' or mode=='validation':
            self.mode='test'
        if self.mode=='train':
            self.batchsize=self.trainbatchsize
        elif self.mode=='valid':
            self.batchsize=self.validbatchsize
        else:
            self.batchsize=self.testbatchsize


    def forward(self,input_headRelIds):
        from scipy import sparse
        # Only one instance (here 1x2 [relid,entid]) gets passed to the fun
        # input_headEntIds is a tensor of batch of selected head and its relation id
        self.relation_current_id=input_headRelIds[0][0].tolist()
        self.head_current_id=input_headRelIds[0][1].tolist()
        self.relation_current_cluster_id= self.relations_clusters_id[self.relation_current_id] 
        self.head_current_cluster_id= self.entities_clusters_id[self.head_current_id] 
        # get embeds for each:
        self.De = self.fromnumpy((np.asarray(self.entities_embeds[self.head_current_id, :])[0]))
        self.Dr = self.fromnumpy((np.asarray(self.relations_embeds[self.relation_current_id, :])[0]))
        # get positive candidates, sample negative candidates: (positively samples all, negatively samples some)
        self.ps=self.tails_glossary_tensor.data[ self.relation_current_id, [self.head_current_id*self.tlen +ii for ii in range(self.tlen)]].todense().cpu().numpy()
        self.psid= self.ps.nonzero()[1]
        negative_cnt= self.ps.shape[1]-np.sum(self.ps!=0)
        probs4neg= np.random.random(self.ps.shape)
        self.ns= (sparse.lil_matrix(np.ones(self.ps.shape))-self.ps).multiply(probs4neg<self.negative_sampler_thresh).tolil()
        self.nsid= self.ns.nonzero()[1]
        # get params for each:
        self.e=self.tnnparameter(self.E[self.head_current_id,:].clone()) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.r=self.tnnparameter(self.R[self.relation_current_id,:].clone()) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.cr=self.tnnparameter(self.CR[ self.relation_current_cluster_id ,:].clone())
        self.ce=self.tnnparameter(self.CE[self.head_current_cluster_id,:].clone())
        # get (clus regul)params for each:
        if self.regularize_within_clusters_relations == True:
            ids4relationsclus=list(set(self.id2clustersid4relations[self.relation_current_cluster_id])-set([self.relation_current_cluster_id]))
            self.clusrelations= self.tnnparameter( self.R[ids4relationsclus,:].clone() )# except for the main entity
            self.clusrelations_plus_mainrelation = t.cat((self.clusrelations,self.r.unsqueeze(0)),dim=0)
            lossreguls1 = self.regularization_rate_relations * t.sqrt(t.sum(self.clusrelations_plus_mainrelation.var(0)))
        else:
            lossreguls1=t.tensor(0)
        if self.regularize_within_clusters_entities == True:
            ids4entitiesclus=list(set(self.id2clustersid4entities[self.head_current_cluster_id])-set([self.head_current_cluster_id]))
            self.clusentities= self.tnnparameter( self.E[ids4entitiesclus,:].clone() ) # except for the main entity
            self.clusentities_plus_mainentity = t.cat((self.clusentities,self.e.unsqueeze(0)),dim=0)
            lossreguls2= self.regularization_rate_entities * t.sqrt(t.sum(self.clusentities_plus_mainentity.var(0)))
        else:
            lossreguls2 = t.tensor(0)
        lossreguls= lossreguls1+lossreguls2
        # pass first layer of projB:
        lyr1= (self.De.unsqueeze(0).mm( self.dimred_e)*(self.e.unsqueeze(0)) + self.ce.unsqueeze(0)).t() * (self.Dr.unsqueeze(0).mm( self.dimred_r)*(self.r.unsqueeze(0)) + self.cr.unsqueeze(0))
        # pass activation function f : here tanh
        lyr1_act=t.tanh(lyr1)
        # pass next layer of projB:
        lyr2_pos = self.fromnumpy(self.entities_embeds[self.psid, :]).mm(self.dimred_e).mm(lyr1_act.mm(self.r.unsqueeze(0).t()) + self.bp.t()) 
        lyr2_neg= self.fromnumpy(self.entities_embeds[self.nsid,:]).mm( self.dimred_e).mm(lyr1_act.mm(self.r.unsqueeze(0).t())+self.bp.t())
        # pass activation function g: here softmax
        if self.projElossfcnType == 'pointwise':
            lyr2_pos_act= self.projBfcntype(lyr2_pos)
            lyr2_neg_act= self.projBfcntype(lyr2_neg)
        else:
            lyr2_pos_act= self.projBfcntype(lyr2_pos,0)
            lyr2_neg_act= self.projBfcntype(lyr2_neg,0)
        # pass to margin-based loss function  # get cross entropy with its corresponding tails as outputs (supervised weighted labels)
        if self.mode == 'train':
            vec_of_relids_rep_neg= [self.relation_current_id for _ in range(len(self.nsid))]
            lossneg= -self.fromnumpy(self.ns[0,self.nsid]/(1+self.sumoverheads[vec_of_relids_rep_neg, self.nsid])/(1+self.sumovertails[self.relation_current_id, self.head_current_id]) ).mm(t.log(1-lyr2_neg_act+1e-9))
            vec_of_relids_rep_pos= [self.relation_current_id for _ in range(len(self.psid))]
            losspos= -self.fromnumpy(self.ps[0,self.psid]/(1+self.sumoverheads[vec_of_relids_rep_pos, self.psid])/(1+self.sumovertails[self.relation_current_id, self.head_current_id]) ).mm(t.log(lyr2_pos_act+1e-9))
            loss=lossneg+losspos+lossreguls
            # redefine the optimizer:
            self.opt=t.optim.SGD(self.parameters(),lr=0.05,momentum=0.3,weight_decay=self.weight_decay) ##TODO: Add args' tunable arguments
            self.opt.zero_grad()
            # backward the loss to compute gradients
            loss.backward()
            # update the data :
            self.update_entities_relations_parameters(self.head_current_id,self.relation_current_id,self.head_current_cluster_id,self.relation_current_cluster_id,ids4relationsclus,ids4entitiesclus)
            # clear unneeded things:
            ##TODO
        elif self.mode=='test':
            vec_of_relids_rep_neg= [self.relation_current_id for _ in range(len(self.nsid))]
            lossneg= -self.fromnumpy(self.ns[0,self.nsid]/(1+self.sumoverheads[vec_of_relids_rep_neg, self.nsid]) ).mm(t.log(1-lyr2_neg_act+1e-9))
            vec_of_relids_rep_pos= [self.relation_current_id for _ in range(len(self.psid))]
            losspos= -self.fromnumpy(self.ps[0,self.psid]/(1+self.sumoverheads[vec_of_relids_rep_pos, self.psid]) ).mm(t.log(lyr2_pos_act+1e-9))
            pass
        if self.cudaenabled==False:
            return [lossneg.data.numpy().squeeze(0).squeeze(0).tolist(),losspos.data.numpy().squeeze(0).squeeze(0).tolist(),lossreguls.data.numpy().squeeze(0).squeeze(0).tolist()] 
        else:
            return [lossneg.data.cpu().numpy().squeeze(0).squeeze(0).tolist(),losspos.data.cpu().numpy().squeeze(0).squeeze(0).tolist(),lossreguls.data.cpu().numpy().squeeze(0).squeeze(0).tolist()] 






    def forward_batch(self,input_headRelIds):
        from scipy import sparse
        # Only one instance (here 1x2 [relid,entid]) gets passed to the fun
        # input_headEntIds is a tensor of batch of selected head and its relation id


        input_headRelIds=np.asarray(input_headRelIds)
        self.relations_current_id=input_headRelIds[:,0].tolist()
        self.heads_current_id=input_headRelIds[:,1].tolist()
        self.relations_current_cluster_id= self.relations_clusters_id[self.relations_current_id]  ##TODO: TEST IT VOCAB RIGHT OR WRONG
        self.heads_current_cluster_id= self.entities_clusters_id[self.heads_current_id]  ##TODO: TEST IT VOCAB RIGHT OR WRONG
        # get embeds for each:
        self.De = self.fromnumpy(self.entities_embeds[self.heads_current_id, :]) #self.De = self.entities_embeds[self.heads_current_id, :]batch x heads
        self.Dr = self.fromnumpy(self.relations_embeds[self.relations_current_id, :])#self.Dr = self.relations_embeds[self.relations_current_id, :] batch x rels
        # get positive candidates, sample negative candidates: (positively samples all, negatively samples some)
        ## get rels , heads and tails vecs ,, then save them FOR NEXT, combine head and tail to get 2nd indice of tnsr
        rels,heads,tails,tailsid,tailsneg,tailsnegid,tailsnegsumover,relsneg=[],[],[],[],[],[],[],[]
        for iu,(rel,head) in enumerate(input_headRelIds):
            locs=[head*self.tlen+iu for iu in range(self.tlen)]
            alltail=self.tails_glossary_tensor.data[rel, locs].nonzero()[1].tolist()
            alltailneg = np.asarray(list(set(list(range(self.tlen)))-set(alltail)))
            alltailneg = alltailneg[np.random.permutation(len(alltailneg))[:int(np.floor(self.negative_sampler_thresh * len(alltailneg)))]]
            tailsneg.extend(alltailneg)
            tailsnegid.extend([iu] * len(alltailneg))
            relsneg.extend([rel for _ in range(len(alltailneg))])
            rels.extend([rel for _ in range(len(alltail))])
            heads.extend([head for _ in range(len(alltail))])
            tails.extend(alltail)
            tailsid.extend([iu]*len(alltail))
        tailsnegsumover=[len(tailsneg)] * len(tailsneg)
        rels=np.asarray(rels)
        heads=np.asarray(heads)
        tails=np.asarray(tails)
        dim2=heads*self.tlen + tails
        psid=tails#[rels,heads]
        tmp_ps=self.tails_glossary_tensor.data[rels,dim2].todense()
        if self.normalize_candidate_tails==True and self.normalize_candidate_heads==True :#and self.mode=='train' :
            ps4test= self.fromnumpy(tmp_ps)/(1e-9 +self.fromnumpy( self.sumovertails[rels, heads]))
            ps=ps4test/(1e-9 +self.fromnumpy( self.sumoverheads[rels, psid]))
        elif self.normalize_candidate_tails == True :
            ps4test= self.fromnumpy(tmp_ps)/(1e-9 + self.fromnumpy( self.sumovertails[rels, heads] ))
            ps = ps4test
        elif self.normalize_candidate_heads == True:# and self.mode=='train' :
            ps4test= self.fromnumpy(tmp_ps)
            ps = ps4test/(1e-9 + self.fromnumpy( self.sumoverheads[rels, psid]))
        else:
            ps4test = self.fromnumpy(tmp_ps)
            ps = ps4test

        #
        nsid=tailsneg
        tmp_ns=np.ones((1,len(nsid)))
        if self.normalize_candidate_tails==True and self.normalize_candidate_heads==True :#and self.mode=='train'  :
            ns4test= self.fromnumpy(tmp_ns)/self.fromnumpy(np.asarray(tailsnegsumover).astype(float))
            ns=ns4test / (1 +self.fromnumpy( self.sumoverheads[relsneg, nsid]))
        elif self.normalize_candidate_tails == True :
            ns4test= self.fromnumpy(tmp_ns)/self.fromnumpy(np.asarray(tailsnegsumover).astype(float))
            ns = ns4test
        elif self.normalize_candidate_heads == True:# and self.mode=='train' :
            ns4test= self.fromnumpy(tmp_ns)
            ns = ns4test/(1 +self.fromnumpy(self.sumoverheads[rels, nsid]))
        else:
            ns4test = self.fromnumpy(tmp_ns)
            ns = ns4test
        # print('negative_positive_len now %1d %1d'%(len(ps),len(ns))) #$#

        # get params for each:
        self.e=self.tnnparameter(self.E[self.heads_current_id,:].clone()) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.r=self.tnnparameter(self.R[self.relations_current_id,:].clone()) # ALSO THIS SUSTAINS OBJ w1.data -= learning_rate * w1.grad.data
        self.cr=self.tnnparameter(self.CR[ self.relations_current_cluster_id ,:].clone())
        self.ce=self.tnnparameter(self.CE[self.heads_current_cluster_id,:].clone())
        # get (clus regul)params for each:
        self.relationids_cluster, self.entityids_cluster, self.relationids_cluster_len, self.entityids_cluster_len, self.relationids_cluster_belongs, self.entityids_cluster_belongs=[],[], [], [], [], []
        xind,xval=[],[]
        self.relationids_cluster=[]
        if self.regularize_within_clusters_relations == True:
            for num,id in enumerate(self.relations_current_cluster_id):
                ids4relationsclus=self.id2clustersid4relations[id]
                self.relationids_cluster.extend(ids4relationsclus)
                self.relationids_cluster_belongs.extend([id]*len(ids4relationsclus))
                self.relationids_cluster_len.append(len(ids4relationsclus))
                xind.extend([num]*len(ids4relationsclus))
                xval.extend([1/len(ids4relationsclus)]*len(ids4relationsclus))
            unified_relationids_cluster, indicesOf_relationids_cluster = np.unique(self.relationids_cluster,return_inverse=True)
            main_unified_relationids_locs = [ii for ii,itm in enumerate(unified_relationids_cluster) if itm in self.relations_current_id]
            other_unified_relationids_locs= list(set(range(len(unified_relationids_cluster)))-set(main_unified_relationids_locs) )
            # create a map todeal with same kinds easier:
            relocate_mapper=np.zeros(len(unified_relationids_cluster)).astype(int)
            relocate_mapper[[*main_unified_relationids_locs,*other_unified_relationids_locs]]= np.asarray(range(len(unified_relationids_cluster)))
            # now transform indicesOf_relationids_cluster by the mapper:
            unified_relationids_cluster_new=unified_relationids_cluster[[*main_unified_relationids_locs,*other_unified_relationids_locs]]
            indicesOf_relationids_cluster_new = relocate_mapper[indicesOf_relationids_cluster]
            # define new parameter matrix:
            idswhere2loc_relationsclus= unified_relationids_cluster[other_unified_relationids_locs].tolist()
            self.clusrelations_other= self.tnnparameter( self.R[idswhere2loc_relationsclus,:].clone() )# except for the main entity
            clusrelations_plus_mainrelation_unified = t.cat((self.r, self.clusrelations_other) ,dim=0)
            clusrelations=clusrelations_plus_mainrelation_unified[indicesOf_relationids_cluster_new,:]
            clusteridmatrix= np.zeros((len(self.relations_current_cluster_id),clusrelations.shape[0]))
            clusteridmatrix[xind,list(range(len(xind)))]=xval
            clusteridmatrix_t=self.fromnumpy(clusteridmatrix)
            lossreguls1 = self.regularization_rate_relations * t.sum( t.sqrt(clusteridmatrix_t.mm(clusrelations**2) - clusteridmatrix_t.mm(clusrelations)**2 ))
        else:
            lossreguls1=t.tensor(0)
        ##
        ##
        ##
        xind, xval = [], []
        self.entityids_cluster=[]
        if self.regularize_within_clusters_entities == True:
            for num,id in enumerate(self.heads_current_cluster_id):
                ids4entitiesclus=self.id2clustersid4entities[id]
                self.entityids_cluster.extend(ids4entitiesclus)
                self.entityids_cluster_belongs.extend([id]*len(ids4entitiesclus))
                self.entityids_cluster_len.append(len(ids4entitiesclus))
                xind.extend([num]*len(ids4entitiesclus))
                xval.extend([1/len(ids4entitiesclus)]*len(ids4entitiesclus))
            unified_entityids_cluster, indicesOf_entityids_cluster = np.unique(self.entityids_cluster,return_inverse=True)
            main_unified_entityids_locs = [ii for ii,itm in enumerate(unified_entityids_cluster) if itm in self.heads_current_id]
            other_unified_entityids_locs= list(set(range(len(unified_entityids_cluster)))-set(main_unified_entityids_locs) )
            # create a map todeal with same kinds easier:
            relocate_mapper=np.zeros(len(unified_entityids_cluster)).astype(int)
            relocate_mapper[[*main_unified_entityids_locs,*other_unified_entityids_locs]]= np.asarray(range(len(unified_entityids_cluster)))
            # now transform indicesOf_entityids_cluster by the mapper:
            unified_entityids_cluster_new=unified_entityids_cluster[[*main_unified_entityids_locs,*other_unified_entityids_locs]]
            indicesOf_entityids_cluster_new = relocate_mapper[indicesOf_entityids_cluster]
            # define new parameter matrix:
            idswhere2loc_entitiesclus=unified_entityids_cluster[other_unified_entityids_locs].tolist()
            self.clusentities_other= self.tnnparameter( self.E[idswhere2loc_entitiesclus,:].clone() )# except for the main entity
            clusentities_plus_mainentity_unified = t.cat((self.e, self.clusentities_other) ,dim=0)
            clusentities=clusentities_plus_mainentity_unified[indicesOf_entityids_cluster_new,:]
            clusteridmatrix= np.zeros((len(self.heads_current_cluster_id),clusentities.shape[0]))
            clusteridmatrix[xind,list(range(len(xind)))]=xval
            clusteridmatrix_t=self.fromnumpy(clusteridmatrix)
            lossreguls2= self.regularization_rate_entities * t.sum( t.sqrt(clusteridmatrix_t.mm(clusentities**2) - clusteridmatrix_t.mm(clusentities)**2 ))
        else:
            lossreguls2=t.tensor(0)
        lossreguls= lossreguls1+lossreguls2
        # pass first layer of projB:
        #lyr1= (self.De.unsqueeze(0).mm( self.dimred_e)*(self.e.unsqueeze(0)) + self.ce.unsqueeze(0)).t() * (self.Dr.unsqueeze(0).mm( self.dimred_r)*(self.r.unsqueeze(0)) + self.cr.unsqueeze(0))
        lyr1= ((self.De.mm( self.dimred_e)*self.e + self.ce).unsqueeze(2).repeat((1,1,self.reldim))) * ((self.Dr.mm( self.dimred_r)*self.r + self.cr).unsqueeze(2).repeat((1,1,self.entdim)).transpose(2,1))
        # pass activation function f : here tanh
        lyr1_act=t.tanh(lyr1) # 3d output batch x ents x rels
        # pass next layer of projB:
        lyr1_act_rsimplified= (lyr1_act * self.r.unsqueeze(1).repeat((1,self.entdim,1)) ).sum(2) + self.bp.repeat((self.batchsize,1,1)).squeeze(1) #  2d output batch x ents
        lyr1_act_rsimplified_replicatedalongbatches= lyr1_act_rsimplified[tailsid,:]
        lyr1_act_rsimplified_replicatedalongbatches_neg= lyr1_act_rsimplified[tailsnegid,:]
        ## positive phase:
        tailsbatch_embeds_dimreduced_pos=self.fromnumpy(self.entities_embeds[psid,:]).mm( self.dimred_e)
        lyr2_pos = (tailsbatch_embeds_dimreduced_pos * lyr1_act_rsimplified_replicatedalongbatches).sum(1)
        ## negative phase:
        tailsbatch_embeds_dimreduced_neg=self.fromnumpy(self.entities_embeds[nsid,:]).mm( self.dimred_e)
        lyr2_neg = (tailsbatch_embeds_dimreduced_neg * lyr1_act_rsimplified_replicatedalongbatches_neg).sum(1)
        # pass activation function g: here softmax or sigmoid for listwise or pointwise .But DUe to the fact that sigmoid must be implemented samplewise, you have to reformulate:
        if self.projElossfcnType == 'pointwise':
            lyr2_pos_act= self.projBfcntype(lyr2_pos)
            lyr2_neg_act= self.projBfcntype(lyr2_neg)
        elif self.projElossfcnType == 'listwise_sumoverbatch':
            lyr2_pos_act = self.projBfcntype(lyr2_pos,0)
            lyr2_neg_act = self.projBfcntype(lyr2_neg,0)
        else: # for listwise case u have to compute numer and denom manually
            numerator_softmax_pos= lyr2_pos.exp()
            numerator_softmax_pos=numerator_softmax_pos/numerator_softmax_pos.norm(1)
            matrix4sum= np.zeros((len(numerator_softmax_pos),len(numerator_softmax_pos)))
            matrix4sum[list(range(len(tailsid))),tailsid]=1
            denomerator_softmax_pos= numerator_softmax_pos.unsqueeze(0).mm(self.fromnumpy(matrix4sum.astype(float)))
            lyr2_pos_act= numerator_softmax_pos/denomerator_softmax_pos
            numerator_softmax_neg= lyr2_neg.exp()
            numerator_softmax_neg=numerator_softmax_neg/numerator_softmax_neg.norm(1)
            matrix4sum= np.zeros((len(numerator_softmax_neg),len(numerator_softmax_neg)))
            matrix4sum[list(range(len(tailsnegid))),tailsnegid]=1
            denomerator_softmax_neg= numerator_softmax_neg.unsqueeze(0).mm(self.fromnumpy(matrix4sum.astype(float)))
            lyr2_neg_act= numerator_softmax_neg/denomerator_softmax_neg
        # FInal Loss function:        # pass to margin-based loss function  # get cross entropy with its corresponding tails as outputs (supervised weighted labels)
        lossposs0_log= (t.log(lyr2_pos_act.unsqueeze(1)+1e-9))
        lossposs0=-ps.t()*lossposs0_log
        losspos=lossposs0.sum()
        lossregulsnp=lossreguls.data.cpu().numpy().squeeze(0).squeeze(0).tolist()
        #
        lossnegs0_profile = -ns4test.t()* (t.log(lyr2_neg_act.unsqueeze(1) + 1e-9))
        lossneg_profile = lossnegs0_profile.sum()
        lossposs0_profile = -ps4test.t()* lossposs0_log
        lossposs_profile = lossposs0_profile.sum()
        lossnegnp_profile, lossposnp_profile=lossneg_profile.data.cpu().numpy().squeeze(0).squeeze(0).tolist(), lossposs_profile.data.cpu().numpy().squeeze(0).squeeze(0).tolist()
        if self.mode == 'train':
            lossnegs0 = -ns.t() * (t.log(1 - lyr2_neg_act.unsqueeze(1) + 1e-9))
            lossneg = lossnegs0.sum()
            loss=lossneg+losspos+lossreguls
            lossnegnp, lossposnp=lossneg.data.cpu().numpy().squeeze(0).squeeze(0).tolist(), losspos.data.cpu().numpy().squeeze(0).squeeze(0).tolist()
            self.opt=t.optim.SGD(self.parameters(),lr=self.learningrate,momentum=self.momentum,weight_decay=self.weight_decay) ##TODO: Add args' tunable arguments
            self.opt.zero_grad()
            # backward the loss to compute gradients
            loss.backward()
            # update the data :
            # self.update_entities_relations_parameters_batch(self.heads_current_id,self.relations_current_id,self.heads_current_cluster_id,self.relations_current_cluster_id, idswhere2loc_relationsclus, idswhere2loc_entitiesclus)
            self.update_entities_relations_parameters_batch_woclus(self.heads_current_id,self.relations_current_id,self.heads_current_cluster_id,self.relations_current_cluster_id)
            # update best and worst results:
            self.average_batch_loss_vector.extend((-lossnegs0).data.cpu().squeeze(1).numpy().tolist())
            self.average_batch_loss_vector.extend((-lossposs0).data.cpu().squeeze(1).numpy().tolist())
            self.average_batch_loss_vector.sort()
            self.average_batch_loss_vector=[*self.average_batch_loss_vector[:self.batchsize],*self.average_batch_loss_vector[-self.batchsize:]]
            # clear unneeded things:
            ##TODO
            # return [lossnegnp,lossposnp,lossregulsnp] #[loss.data.cpu().numpy().squeeze(0).squeeze(0).tolist()]
        # else: #'test' 'valid'
        # Binary test whether model has detected rightly:
        hasdetected= lossposnp_profile<lossnegnp_profile
        # Quantitative test to see how much precise the model acted out:
        detectionscore= lossnegnp_profile/(lossposnp_profile+lossnegnp_profile)
        numoftailsinthisscore= lossposs0.shape[0]
        return [lossnegnp_profile,lossposnp_profile,lossregulsnp,hasdetected,detectionscore,numoftailsinthisscore] #[loss.data.cpu().numpy().squeeze(0).squeeze(0).tolist()]








    def update_entities_relations_parameters_batch_woclus(self,entid,relid,entclid,relclid):
        ##note: self.dimred_e self.dimred_r are automatically retained and learned and don't need this func
        # once grads are available, the parameters get updated:
        self.opt.step()
        # now all the changed data gets resaved back to the glossaries:
        # if self.cudaenabled==False:
        # self.E.data[ids4entitiesclus,:]=self.clusentities_other.data # entity vecs to indirectly learn
        # self.R.data[ids4relationsclus,:]=self.clusrelations_other.data # relation vecs to indirectly learn
        self.E.data[entid,:]=self.e.data # entity vecs to indirectly learn
        self.R.data[relid,:]=self.r.data # relation vecs to indirectly learn
        self.CE.data[entclid,:]=self.ce.data # Parameters of all ENTITIES' clusters to indirectly learn
        self.CR.data[relclid,:]=self.cr.data # Parameters of all RELATIONS' clusters to indirectly learn





    def update_entities_relations_parameters_batch(self,entid,relid,entclid,relclid,ids4relationsclus,ids4entitiesclus):
        ##note: self.dimred_e self.dimred_r are automatically retained and learned and don't need this func
        # once grads are available, the parameters get updated:
        self.opt.step()
        # now all the changed data gets resaved back to the glossaries:
        # if self.cudaenabled==False:
        self.E.data[ids4entitiesclus,:]=self.clusentities_other.data # entity vecs to indirectly learn
        self.R.data[ids4relationsclus,:]=self.clusrelations_other.data # relation vecs to indirectly learn
        self.E.data[entid,:]=self.e.data # entity vecs to indirectly learn
        self.R.data[relid,:]=self.r.data # relation vecs to indirectly learn
        self.CE.data[entclid,:]=self.ce.data # Parameters of all ENTITIES' clusters to indirectly learn
        self.CR.data[relclid,:]=self.cr.data # Parameters of all RELATIONS' clusters to indirectly learn



    def update_entities_relations_parameters(self,entid,relid,entclid,relclid,ids4relationsclus,ids4entitiesclus):
        ##note: self.dimred_e self.dimred_r are automatically retained and learned and don't need this func
        # once grads are available, the parameters get updated:
        self.opt.step()
        # now all the changed data gets resaved back to the glossaries:
        # if self.cudaenabled==False:
        self.E.data[ids4entitiesclus,:]=self.clusentities.data # entity vecs to indirectly learn
        self.R.data[ids4relationsclus,:]=self.clusrelations.data # relation vecs to indirectly learn
        self.E.data[entid,:]=self.e.data # entity vecs to indirectly learn
        self.R.data[relid,:]=self.r.data # relation vecs to indirectly learn
        self.CE.data[entclid,:]=self.ce.data # Parameters of all ENTITIES' clusters to indirectly learn
        self.CR.data[relclid,:]=self.cr.data # Parameters of all RELATIONS' clusters to indirectly learn
