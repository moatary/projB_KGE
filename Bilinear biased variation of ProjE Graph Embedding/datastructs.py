from scipy.sparse import lil_matrix
from scipy import sparse
import numpy as np


'''
class tens3d():
    from scipy import sparse
    #from tools_metrics import nest_weight
    def __init__(self,size):
        self.hlen = size[0]
        self.rlen= size[1]
        self.tlen= size[2]
        self.data = [None] * self.rlen
        self.tensgen=lambda : sparse.lil_matrix((self.hlen , self.tlen ))

    def __getitem__(self, item):
        print(item)
        # if type(item)==tuple:
        #     if self.data[item[1]]==None:
        #         self.data[item[1]]= self.tensgen()
        #     return self.data[item[1]][item[0],item[2]]
        # else:
        #     if self.data[item]==None:
        #         self.data[item]= self.tensgen()
        #     return self.data[item]


    def __setitem__(self, key, value): #WARNING # this setting does not actually set. It increases by a weight.
        # if len(key)!=3:
        #     raise NotImplementedError
        # else:
        if self.data[key[1]]==None:
            self.data[key[1]]= self.tensgen()
        self.data[key[1]][key[0],key[2]]+=value
'''


def to1hot(vec,length=None):
    import numpy as np
    if length==None:
        length=len(vec)
    return np.sum(np.eye(length)[vec],1)

class tensor3d():
    # from datastructs import where
    '''
    objective is to define a tensor with back-end of a sparse 2d matrix
    # when enabling default dim, the chosen dim automatically comes first to be accessed easier way. In matrix mode, the role changes
    '''
    #from tools_metrics import nest_weight
    def __init__(self,size=[None,None,None], defaultdim=1, vectormodereturn=1):
        self.hlen = size[0]
        self.rlen= size[1]
        self.tlen= size[2]
        self.vectormodereturn=vectormodereturn
        self.defaultdim=defaultdim
        if size[0] is not None and size[1] is not None and size[2] is not None:
            if self.vectormodereturn==1:
                if self.defaultdim == 1: # for relationcluster/relation pred...
                    self.data = sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                elif self.defaultdim == 0:  # for headclustering
                    self.data = sparse.lil_matrix((self.hlen, self.rlen * self.tlen))
                elif self.defaultdim == 2: # for tailclustering
                    self.data = sparse.lil_matrix((self.tlen, self.hlen * self.rlen))
                if self.defaultdim == -1:  # for relationcluster/relation pred...
                    self.data = sparse.lil_matrix((self.rlen, self.tlen * self.hlen))
                elif self.defaultdim == -2:  # for tailclustering
                    self.data = sparse.lil_matrix((self.tlen, self.rlen * self.hlen))
            else: # matrix mode: u have to combine defaultdim with one side of mtx to easily extract data wo rshaping and copying
                if self.defaultdim == 1: # for relationcluster/relation pred...
                    self.data = sparse.lil_matrix((self.hlen*self.rlen , self.tlen ))
                elif self.defaultdim == 0:  # for headclustering
                    self.data = sparse.lil_matrix((self.hlen*self.rlen , self.tlen))
                elif self.defaultdim == 2: # for tailclustering
                    self.data = sparse.lil_matrix((self.hlen, self.rlen * self.tlen))
                if self.defaultdim == -1: # for relationcluster/relation pred...
                    self.data = sparse.lil_matrix((self.hlen*self.tlen , self.rlen ))
                elif self.defaultdim == -2: # for tailclustering
                    self.data = sparse.lil_matrix((self.hlen, self.tlen * self.rlen))
        self.where= where(self)

    def load(cls,data, defaultdim=1, vectormodereturn=1, mutate=0):
        import pickle
        import numpy as np
        from scipy import sparse
        ##TODO : Not debugged
        # for the case of low memory condition, just replace whole object with the input :mutate=1
        if type(data) is list:
            datasize= [len(data),len(data[0])]
        elif type(data) is np:
            datasize= data.shape
        else:
            mutate=2
        # else:
        #     raise NotImplementedError
        self=tensor3d()
        self.vectormodereturn=vectormodereturn
        self.defaultdim=defaultdim
        ## for the case of low memory condition, just replace whole object with the input :mutate=1
        if mutate == 0: #
            if self.vectormodereturn==1:
                if self.defaultdim == 1: # for relationcluster/relation pred...
                    self.data = sparse.csr_matrix(np.asarray(data).transpose((1,0,2)).reshape((self.rlen , self.hlen*self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                elif self.defaultdim == 0:  # for headclustering
                    self.data = sparse.csr_matrix(np.asarray(data).reshape((self.hlen , self.rlen*self.tlen )))#sparse.lil_matrix((self.hlen, self.rlen * self.tlen))
                elif self.defaultdim == 2: # for tailclustering
                    self.data = sparse.csr_matrix(np.asarray(data).transpose((2,0,1)).reshape((self.tlen , self.hlen*self.rlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            else: # matrix mode: u have to combine defaultdim with one side of mtx to easily extract data wo rshaping and copying
                if self.defaultdim == 1: # for relationcluster/relation pred...
                    self.data = sparse.csr_matrix(np.asarray(data).reshape((self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                elif self.defaultdim == 0:  # for headclustering
                    self.data = sparse.csr_matrix(np.asarray(data).reshape((self.hlen*self.rlen , self.tlen)))#sparse.lil_matrix((self.hlen, self.rlen * self.tlen))
                elif self.defaultdim == 2: # for tailclustering
                    self.data = sparse.csr_matrix(np.asarray(data).reshape((self.hlen, self.rlen * self.tlen))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            return self
        elif mutate==1:# for the case of low memory
            if self.vectormodereturn==1:
                if self.defaultdim == 1: # for relationcluster/relation pred...
                    data = sparse.csr_matrix(np.asarray(data).transpose((1,0,2)).reshape((self.rlen , self.hlen*self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                elif self.defaultdim == 0:  # for headclustering
                    data = sparse.csr_matrix(np.asarray(data).reshape((self.hlen , self.rlen*self.tlen )))#sparse.lil_matrix((self.hlen, self.rlen * self.tlen))
                elif self.defaultdim == 2: # for tailclustering
                    data = sparse.csr_matrix(np.asarray(data).transpose((2,0,1)).reshape((self.tlen , self.hlen*self.rlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            else: # matrix mode: u have to combine defaultdim with one side of mtx to easily extract data wo rshaping and copying
                if self.defaultdim == 1: # for relationcluster/relation pred...
                    data = sparse.csr_matrix(np.asarray(data).reshape((self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                elif self.defaultdim == 0:  # for headclustering
                    data = sparse.csr_matrix(np.asarray(data).reshape((self.hlen*self.rlen , self.tlen)))#sparse.lil_matrix((self.hlen, self.rlen * self.tlen))
                elif self.defaultdim == 2: # for tailclustering
                    data = sparse.csr_matrix(np.asarray(data).reshape((self.hlen, self.rlen * self.tlen))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            self.data= data

        #
        if type(data)==str:
            print(data)
            with open(data,'rb') as ff:
                data=pickle.load(ff)
                print(type(data))
                print(type(data)==type(self))
                if type(data)==sparse.lil_matrix:
                    print('issparse')
                    self.data=data
                    if self.defaultdim == 1:  # for relationcluster/relation pred...
                        self.rlen = data.shape[0]
                        self.hlen = None
                        self.tlen = None
                    elif self.defaultdim == 0:  # for headclustering
                        self.hlen = data.shape[0]
                        self.rlen = None
                        self.tlen = None
                    elif self.defaultdim == 2:  # for tailclustering
                        self.tlen = data.shape[0]
                        self.hlen = None
                        self.rlen = None
                elif type(data.data)==sparse.lil_matrix:
                    print('hr')
                    self.__dict__.update(data.__dict__)
        self.where = where(self)
        return self

    def sumsparsematrix(cls,len1, len2):
        import numpy as np
        from scipy import sparse
        offsets = np.array([x * len2 for x in range(len1)])
        N = len2
        M = len1 * len2
        vals = np.ones(offsets.shape)
        dupvals = np.concatenate((vals, vals[1::]))
        dupoffsets = np.concatenate((offsets, -offsets[1:]))
        return sparse.diags(dupvals, dupoffsets, shape=(M, M), format='lil', dtype=float)[:, :N]

    def sumover(self,what='tails'):
        from scipy.linalg import kron
        if what=='all':
            # first head:
            raise NotImplementedError
        elif what=='heads' or what=='head':
            if self.vectormodereturn==1:
                if self.defaultdim==0:
                    raise NotImplementedError
                elif self.defaultdim==1:
                    return self.data * sparse.kron(np.ones((self.hlen, 1)), np.eye(self.tlen))#self.sumsparsematrix(self.hlen,self.tlen)
                elif self.defaultdim==2:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif what=='tails' or what=='tail':
            if self.vectormodereturn==1:
                if self.defaultdim==0:
                    raise NotImplementedError
                elif self.defaultdim==1:
                    return self.data * sparse.kron(np.eye(self.hlen), np.ones((self.tlen, 1)))
                elif self.defaultdim==2:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif what=='rels' or what=='rel':
            raise NotImplementedError


    def change_defaultdim_to(self,towhat): ##TODO
        raise NotImplementedError

    def change_vectormode(self,isenabled): ##TODO
        raise NotImplementedError


    def __getitem__(self, item):
        if type(item) is not tuple:
            if self.vectormodereturn == 1:
                return self.data[item,:]
            else:
                if type(item) is not list:
                    if self.defaultdim == 1: # for relationcluster/relation pred...
                        requiredindices_dim0 = [ it*self.rlen+item for it in range(self.hlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == 0:  # for headclustering
                        requiredindices_dim0 = [ item*self.rlen+it for it in range(self.rlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == 2: # for tailclustering
                        requiredindices_dim0 = [ it*self.tlen+item for it in range(self.rlen)]
                        return self.data[ :, requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    if self.defaultdim == -1: # for relationcluster/relation pred...
                        requiredindices_dim0 = [ item*self.hlen+it for it in range(self.hlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == -2: # for tailclustering
                        requiredindices_dim0 = [ item*self.rlen+it for it in range(self.rlen)]
                        return self.data[ :, requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))

                else:
                    if self.defaultdim == 1: # for relationcluster/relation pred...
                        requiredindices_dim0 = [ it*self.rlen+itm for itm in item for it in range(self.hlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == 0:  # for headclustering
                        requiredindices_dim0 = [ itm*self.rlen+it for itm in item for it in range(self.rlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == 2: # for tailclustering
                        requiredindices_dim0 = [ it*self.tlen+itm for itm in item for it in range(self.rlen)]
                        return self.data[ :, requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    if self.defaultdim == -1: # for relationcluster/relation pred...
                        requiredindices_dim0 = [ itm*self.hlen+it for itm in item for it in range(self.hlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == -2: # for tailclustering
                        requiredindices_dim0 = [ itm*self.rlen+it for itm in item for it in range(self.rlen)]
                        return self.data[ :, requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        elif type(item[0])==slice and type(item[1])==int and type(item[2])==int:
            if self.defaultdim == 0:
                # NOTE: NAGHES #todo
                return self.data[:,item[1]*self.tlen+item[2]]
            elif self.defaultdim== 1:
                it= [x*self.tlen+item[2] for x in range(self.hlen)]
                return self.data[[item[1] for _ in range(len(it))] ,it]
            elif self.defaultdim==2:
                it= [x*self.rlen+item[1] for x in range(self.hlen)]
                return self.data[[item[2] for _ in range(len(it))],it]

        elif type(item[0]) == int and type(item[1]) == slice and type(item[2]) == int:
            if self.defaultdim==0:
                # NOTE: NAGHES #todo
                items2nd= [x*self.tlen+item[2] for x in range(self.rlen)]
                return self.data[[item[0] for _ in len(items2nd)] ,items2nd]
            if self.defaultdim ==1:
                # NOTE: NAGHES #todo
                items2nd = [x * self.tlen + item[2] for x in range(self.rlen)]
                return self.data[:, item[0]*self.tlen+item[2]]
            if self.defaultdim == 2:
                # NOTE: NAGHES #todo
                items2nd = [x * self.rlen + item[1] for x in range(self.hlen)]
                return self.data[[item[2] for _ in range( len(items2nd))]  , items2nd]
        elif type(item[0]) == int and type(item[1]) == int and type(item[2]) == slice:
            if self.defaultdim == 0:
                # NOTE: NAGHES #todo
                items2nd= [item[1]*self.tlen+x for x in range(self.tlen)]
                return self.data[[item[0] for _ in range(len(items2nd))] , items2nd]
            if self.defaultdim == 1:
                # NOTE: NAGHES #todo
                items2nd = [item[0] * self.tlen + x for x in range(self.tlen)]
                return self.data[[item[1] for _ in range(len(items2nd))], items2nd]
            if self.defaultdim == 2:
                print('ishere')
                # NOTE: NAGHES #todo
                print( item[0]*self.rlen+item[1])
                return self.data[:, item[0]*self.rlen+item[1]]
        elif type(item[0])==int and type(item[1])==list and type(item[2])==list:
            raise NotImplementedError
                    # if self.defaultdim == 1: # for relationcluster/relation pred...
                    #     requiredindices_dim0 = [ it*self.rlen+itm for itm in item for it in range(self.hlen)]
                    #     return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    # elif self.defaultdim == 0:  # for headclustering
                    #     requiredindices_dim0 = [ itm*self.rlen+it for itm in item for it in range(self.rlen)]
                    #     return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    # elif self.defaultdim == 2: # for tailclustering
                    #     requiredindices_dim0 = [ it*self.tlen+itm for itm in item for it in range(self.rlen)]
                    #     return self.data[ :, requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    # if self.defaultdim == -1: # for relationcluster/relation pred...
                    #     requiredindices_dim0 = [ itm*self.hlen+it for itm in item for it in range(self.hlen)]
                    #     return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    # elif self.defaultdim == -2: # for tailclustering
                    #     requiredindices_dim0 = [ itm*self.rlen+it for itm in item for it in range(self.rlen)]
                    #     return self.data[ :, requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        else:
            raise NotImplementedError
            # # item is tuple. so first clarify states of each of three :
            # if len(item)==2:
            #     item = (*item,slice(None,None,None))
            # ### foritem0
            # if type(item[0])==slice:
            #     strt= item[0].start
            #     if strt is None:
            #         strt=0
            #     step= item[0].step
            #     if step is None:
            #         step=1
            #     stop= item[0].stop
            #     if stop is None:
            #         if self.vectormodereturn == 1:
            #             if self.defaultdim == 0:
            #                 stop= self.hlen
            #             elif self.defaultdim== 1:
            #                 stop= self.rlen
            #             else:
            #                 stop= self.tlen
            #         else: #mtx
            #             if self.defaultdim== 0:
            #                 stop= self.hlen
            #             elif self.defaultdim==1:
            #                 stop= self.rlen
            #             else:
            #                 stop= self.hlen
            #     item[0]= list(range(strt,stop,step))
            # ### foritem1
            # if type(item[1])==slice:
            #     strt= item[1].start
            #     if strt is None:
            #         strt=0
            #     step= item[1].step
            #     if step is None:
            #         step=1
            #     stop= item[1].stop
            #     if stop is None:
            #         if self.vectormodereturn == 1:
            #             if self.defaultdim == 0:
            #                 stop= self.rlen
            #             elif self.defaultdim== 1:
            #                 stop= self.hlen
            #             else:
            #                 stop= self.rlen
            #         else: #mtx
            #             if self.defaultdim== 0:
            #                 stop= self.rlen
            #             elif self.defaultdim==1:
            #                 stop= self.hlen
            #             else:
            #                 stop= self.rlen
            #     item[1] = list(range(strt, stop, step))
            # ### foritem2
            # if type(item[2])==slice:
            #     strt= item[2].start
            #     if strt is None:
            #         strt=0
            #     step= item[2].step
            #     if step is None:
            #         step=1
            #     stop= item[2].stop
            #     if stop is None:
            #         if self.vectormodereturn == 1:
            #             if self.defaultdim == 0:
            #                 stop= self.tlen
            #             elif self.defaultdim== 1:
            #                 stop= self.tlen
            #             else:
            #                 stop= self.hlen
            #         else: #mtx
            #             if self.defaultdim== 0:
            #                 stop= self.tlen
            #             elif self.defaultdim==1:
            #                 stop= self.tlen
            #             else:
            #                 stop= self.tlen
            #     item[2] = list(range(strt, stop, step))
            # ### Now all indices are clear. MAin process:
            # if self.vectormodereturn == 1:
            #     # simulation of indices for 2nd and 3rd needed:
            #     if self.defaultdim == 0: # for relationcluster/relation pred...
            #         requiredindices_dim0 = [y*self.tlen+z for y in item[1] for z in item[2] ]
            #         return self.data[ item[0], requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            #     elif self.defaultdim == 1:  # for headclustering
            #         requiredindices_dim0 = [x*self.tlen+z for x in item[0] for z in item[2] ]
            #         return self.data[ item[1], requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            #     elif self.defaultdim == 2: # for tailclustering
            #         requiredindices_dim0 = [x*self.tlen+y for x in item[0] for y in item[1] ]
            #         return self.data[ item[2], requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            # else: # IF that is matrix
            #     if self.defaultdim == 0: # for relationcluster/relation pred...
            #         requiredindices_dim0 = [x*self.tlen+y for x in item[0] for y in item[1] ]
            #         return self.data[  requiredindices_dim0, item[2]]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            #     elif self.defaultdim == 1:  # for headclustering
            #         requiredindices_dim0 = [x*self.tlen+y for x in item[0] for y in item[1] ]
            #         return self.data[ requiredindices_dim0 , item[2]]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            #     elif self.defaultdim == 2: # for tailclustering
            #         requiredindices_dim0 = [y*self.tlen+z for y in item[1] for z in item[2] ]
            #         return self.data[ item[0], requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))





    def __setitem__(self, item, value): #WARNING # this setting does not actually set. It increases by a weight.
         if self.vectormodereturn == 1:
            # simulation of indices for 2nd and 3rd needed:
            if self.defaultdim == 0: # for relationcluster/relation pred...
                requiredindices_dim0 = [y*self.tlen+z for y,z in zip(item[1],item[2])]
                self.data[ item[0], requiredindices_dim0 ]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            elif self.defaultdim == 1:  # for headclustering
                requiredindices_dim0 = [x*self.tlen+z for x,z in zip(item[0],item[2]) ]
                self.data[ item[1], requiredindices_dim0 ]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            elif self.defaultdim == 2: # for tailclustering
                requiredindices_dim0 = [x*self.rlen+y for x,y in zip(item[0],item[1]) ]
                self.data[ item[2], requiredindices_dim0 ]=value   #TODO+=value    #(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            elif self.defaultdim == -1:  # for headclustering
                requiredindices_dim0 = [z*self.hlen+x for x,z in zip(item[0],item[2]) ]
                self.data[ item[1], requiredindices_dim0 ]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            elif self.defaultdim == -2: # for tailclustering
                requiredindices_dim0 = [y*self.hlen+x for x,y in zip(item[0],item[1]) ]
                self.data[ item[2], requiredindices_dim0 ]=value   #TODO+=value    #(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
         else: # IF that is matrix
            if self.defaultdim == 0: # for relationcluster/relation pred...
                requiredindices_dim0 = [x*self.rlen+y for  x,y in zip(item[0],item[1]) ]
                self.data[  requiredindices_dim0, item[2]]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            elif self.defaultdim == 1:  # for headclustering
                requiredindices_dim0 = [x*self.rlen+y for x,y in zip(item[0],item[1]) ]
                self.data[ requiredindices_dim0 , item[2]]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            elif self.defaultdim == 2: # for tailclustering
                requiredindices_dim0 = [y*self.tlen+z for y,z in zip(item[1],item[2]) ]
                self.data[ item[0], requiredindices_dim0 ]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            elif self.defaultdim == -1:  # for headclustering
                requiredindices_dim0 = [y*self.hlen+x for x,y in zip(item[0],item[1]) ]
                self.data[ requiredindices_dim0 , item[2]]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
            elif self.defaultdim == -2: # for tailclustering
                requiredindices_dim0 = [z*self.rlen+y for y,z in zip(item[1],item[2]) ]
                print(np.max(requiredindices_dim0))
                try:
                    self.data[ item[0], requiredindices_dim0 ]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                except:
                    pass






            #thats just an int. So respond a vec:


    # def __setitem__(self, item, value):
    #     raise NotImplementedError
        # if type(item) is not tuple:
        #     if self.vectormodereturn == 1:
        #         self.data[item, :]= value.reshape((value.shape[0],value.shape[1]*value.shape[2]))
        #     else:
        #         if type(item) is not list:
        #             if self.defaultdim == 1: # for relationcluster/relation pred...
        #                 requiredindices_dim0 = [ it*self.rlen+item for it in range(self.hlen)]
        #                 return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #             elif self.defaultdim == 0:  # for headclustering
        #                 requiredindices_dim0 = [ item*self.rlen+it for it in range(self.hlen)]
        #                 return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #             elif self.defaultdim == 2: # for tailclustering
        #                 requiredindices_dim0 = [ it*self.tlen+item for it in range(self.hlen)]
        #                 return self.data[ :, requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #         else:
        #             if self.defaultdim == 1: # for relationcluster/relation pred...
        #                 requiredindices_dim0 = [ it*self.rlen+itm for itm in item for it in range(self.hlen)]
        #                 return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #             elif self.defaultdim == 0:  # for headclustering
        #                 requiredindices_dim0 = [ itm*self.rlen+it for itm in item for it in range(self.hlen)]
        #                 return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #             elif self.defaultdim == 2: # for tailclustering
        #                 requiredindices_dim0 = [ it*self.tlen+itm for itm in item for it in range(self.hlen)]
        #                 return self.data[ :, requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        # else:
        #     # item is tuple. so first clarify states of each of three :
        #     if len(item)==2:
        #         item = (*item,slice(None,None,None))
        #     ### foritem0
        #     if type(item[0])==slice:
        #         strt= item[0].start
        #         if strt is None:
        #             strt=0
        #         step= item[0].step
        #         if step is None:
        #             step=1
        #         stop= item[0].stop
        #         if stop is None:
        #             if self.vectormodereturn == 1:
        #                 if self.defaultdim == 0:
        #                     stop= self.hlen
        #                 elif self.defaultdim== 1:
        #                     stop= self.rlen
        #                 else:
        #                     stop= self.tlen
        #             else: #mtx
        #                 if self.defaultdim== 0:
        #                     stop= self.hlen
        #                 elif self.defaultdim==1:
        #                     stop= self.rlen
        #                 else:
        #                     stop= self.hlen
        #         item[0]= list(range(strt,stop,step))
        #     ### foritem1
        #     if type(item[1])==slice:
        #         strt= item[1].start
        #         if strt is None:
        #             strt=0
        #         step= item[1].step
        #         if step is None:
        #             step=1
        #         stop= item[1].stop
        #         if stop is None:
        #             if self.vectormodereturn == 1:
        #                 if self.defaultdim == 0:
        #                     stop= self.rlen
        #                 elif self.defaultdim== 1:
        #                     stop= self.hlen
        #                 else:
        #                     stop= self.rlen
        #             else: #mtx
        #                 if self.defaultdim== 0:
        #                     stop= self.rlen
        #                 elif self.defaultdim==1:
        #                     stop= self.hlen
        #                 else:
        #                     stop= self.rlen
        #         item[1] = list(range(strt, stop, step))
        #     ### foritem2
        #     if type(item[2])==slice:
        #         strt= item[2].start
        #         if strt is None:
        #             strt=0
        #         step= item[2].step
        #         if step is None:
        #             step=1
        #         stop= item[2].stop
        #         if stop is None:
        #             if self.vectormodereturn == 1:
        #                 if self.defaultdim == 0:
        #                     stop= self.tlen
        #                 elif self.defaultdim== 1:
        #                     stop= self.tlen
        #                 else:
        #                     stop= self.hlen
        #             else: #mtx
        #                 if self.defaultdim== 0:
        #                     stop= self.tlen
        #                 elif self.defaultdim==1:
        #                     stop= self.tlen
        #                 else:
        #                     stop= self.tlen
        #         item[2] = list(range(strt, stop, step))
        #     ### Now all indices are clear. MAin process:
        #     if self.vectormodereturn == 1:
        #         # simulation of indices for 2nd and 3rd needed:
        #         if self.defaultdim == 0: # for relationcluster/relation pred...
        #             requiredindices_dim0 = [y*self.tlen+z for y in item[1] for z in item[2] ]
        #             return self.data[ item[0], requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #         elif self.defaultdim == 1:  # for headclustering
        #             requiredindices_dim0 = [x*self.tlen+z for x in item[0] for z in item[2] ]
        #             return self.data[ item[1], requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #         elif self.defaultdim == 2: # for tailclustering
        #             requiredindices_dim0 = [x*self.tlen+y for x in item[0] for y in item[1] ]
        #             return self.data[ item[2], requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #     else: # IF that is matrix
        #         if self.defaultdim == 0: # for relationcluster/relation pred...
        #             requiredindices_dim0 = [x*self.tlen+y for x in item[0] for y in item[1] ]
        #             return self.data[  requiredindices_dim0, item[2]]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #         elif self.defaultdim == 1:  # for headclustering
        #             requiredindices_dim0 = [x*self.tlen+y for x in item[0] for y in item[1] ]
        #             return self.data[ requiredindices_dim0 , item[2]]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
        #         elif self.defaultdim == 2: # for tailclustering
        #             requiredindices_dim0 = [y*self.tlen+z for y in item[1] for z in item[2] ]
        #             return self.data[ item[0], requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))



class where(tensor3d):
    def __init__(self,obj):
        self.obj=obj
        pass

    def __getitem__(self, item):
        return np.where((self.obj.__getitem__(item)).toarray()!=0)[1].tolist()




class tensor3d_mode2():
    '''
    objective is to define a tensor with back-end of a sparse 2d matrix
    # when enabling default dim, the chosen dim automatically comes first to be accessed easier way. In matrix mode, the role changes
    '''
    #from tools_metrics import nest_weight
    def __init__(self,size, defaultdim=1, vectormodereturn=1):
        self.hlen = size[0]
        self.rlen= size[1]
        self.tlen= size[2]
        self.vectormodereturn=vectormodereturn
        self.defaultdim=defaultdim
        if self.vectormodereturn==1:
            if self.defaultdim == 1: # for relationcluster/relation pred...
                self.data = sparse.lil_matrix((self.hlen*self.tlen, self.rlen  ))
            elif self.defaultdim == 0:  # for headclustering
                self.data = sparse.lil_matrix((self.rlen * self.tlen, self.hlen  ))
            elif self.defaultdim == 2: # for tailclustering
                self.data = sparse.lil_matrix(( self.hlen * self.rlen, self.tlen ))
        else: # matrix mode: u have to combine defaultdim with one side of mtx to easily extract data wo rshaping and copying
            if self.defaultdim == 1: # for relationcluster/relation pred...
                self.data = sparse.lil_matrix((self.hlen*self.rlen , self.tlen ))
            elif self.defaultdim == 0:  # for headclustering
                self.data = sparse.lil_matrix((self.hlen*self.rlen , self.tlen))
            elif self.defaultdim == 2: # for tailclustering
                self.data = sparse.lil_matrix((self.rlen * self.tlen, self.hlen ))




    def change_defaultdim_to(self,towhat): ##TODO
        raise NotImplementedError

    def change_vectormode(self,isenabled): ##TODO
        raise NotImplementedError


    def __getitem__(self, item):
        if type(item) is not tuple:
            if self.vectormodereturn == 1:
                return self.data[:, item]
            else:
                if type(item) is not list:
                    if self.defaultdim == 1: # for relationcluster/relation pred...
                        requiredindices_dim0 = [ it*self.rlen+item for it in range(self.hlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == 0:  # for headclustering
                        requiredindices_dim0 = [ item*self.rlen+it for it in range(self.hlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == 2: # for tailclustering
                        requiredindices_dim0 = [ it*self.tlen+item for it in range(self.hlen)]
                        return self.data[ :, requiredindices_dim0 ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                else:
                    if self.defaultdim == 1: # for relationcluster/relation pred...
                        requiredindices_dim0 = [ it*self.rlen+itm for itm in item for it in range(self.hlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == 0:  # for headclustering
                        requiredindices_dim0 = [ itm*self.rlen+it for itm in item for it in range(self.rlen)]
                        return self.data[ requiredindices_dim0 , :]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                    elif self.defaultdim == 2: # for tailclustering
                        requiredindices_dim0 = [ it*self.tlen+itm for itm in item for it in range(self.rlen)]
                        return self.data[  requiredindices_dim0,: ]#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))

        else:
            # item is tuple. so first clarify states of each of three :
            if len(item)==3:
                raise NotImplementedError
                # if contained slice : it means i've to use filtering dimwisely
                    #dummy= sparse.lil_matrix(self.data)
                    # first reduce dummy in dims containing slice
                    # then reduce dummy in dims not containing slice by filtering
                # if no slice existed, thats only selection of list of points. use __setitem__ analogy
            else:
                raise NotImplementedError






    def __setitem__(self, item, value): #WARNING # this setting does not actually set. It increases by a weight.
             if self.vectormodereturn == 1:
                # simulation of indices for 2nd and 3rd needed:
                if self.defaultdim == 0: # for relationcluster/relation pred...
                    requiredindices_dim0 = [y*self.tlen+z for y,z in zip(item[1],item[2])]
                    self.data[ requiredindices_dim0,item[0] ]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                elif self.defaultdim == 1:  # for headclustering
                    requiredindices_dim0 = [x*self.tlen+z for  x,z in zip(item[0],item[2]) ]
                    self.data[ requiredindices_dim0  ,item[1]]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                elif self.defaultdim == 2: # for tailclustering
                    requiredindices_dim0 = [x*self.rlen+y for  x,y in zip(item[0],item[1]) ]
                    self.data[ requiredindices_dim0  ,item[2]]=value   #TODO+=value    #(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
             else: # IF that is matrix
                if self.defaultdim == 0: # for relationcluster/relation pred...
                    requiredindices_dim0 = [x*self.rlen+y for  x,y in zip(item[0],item[1]) ]
                    self.data[  requiredindices_dim0, item[2]]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                elif self.defaultdim == 1:  # for headclustering
                    requiredindices_dim0 = [x*self.rlen+y for x,y in zip(item[0],item[1]) ]
                    self.data[ requiredindices_dim0 , item[2]]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))
                elif self.defaultdim == 2: # for tailclustering
                    requiredindices_dim0 = [y*self.tlen+z for y,z in zip(item[1],item[2]) ]
                    self.data[ requiredindices_dim0, item[0] ]=value#(self.hlen*self.rlen , self.tlen ))) #sparse.lil_matrix((self.rlen , self.hlen*self.tlen ))






















# class treatMtxOfTnsAsVec():
#     def __init__(self, tens3dobj):#, whichdims):
#         # super(self,)
#         self.tns= tens3dobj
#         # self.dim1=whichdims[0]
#         # self.dim2= whichdims[1]
#
#     def __getitem__(self, relindx): # numpy vector
#         if type(relindx) is not type(list):
#             return np.asarray(self.tns[relindx].todense().reshape((1,-1))).squeeze()
#         else:
#             for
#
#     def __get__(self):
#         return self.__getitem__(self,range(len(self.tns.data) ))
#
#







class TensorTypeGraph(object): # generate 3D mtx out of graph(triples) ##todo: not mine

    def __init__(self, triple_dat, n_ent, n_rel):
        self.rel2mat = [lil_matrix((n_ent, n_ent)) for _ in range(n_rel)]
        for triple in triple_dat.batch_iter(1, rand_flg=False):
            sub, rel, obj = triple[0]
            self.rel2mat[rel][sub, obj] += 1.0
    #
    def search_obj_id(self, sub, rel):
        return np.where(self.rel2mat[rel][sub].todense() >0)[1]
    #
    def search_sub_id(self, rel, obj):
        return np.where(self.rel2mat[rel][:, obj].todense() >0 )[0]
    #
    @classmethod
    def load_from_raw(cls, data_path, ent_v, rel_v):
        triples = TripletDataset.load(data_path, ent_v, rel_v)
        return TensorTypeGraph(triples, len(ent_v), len(rel_v))

class TripletDataset(object): #get triples of existing database with def format  ##todo: not fully mine
    def __init__(self, samples):
        assert type(samples) == list or type(samples) == np.ndarray
        self._samples = samples if type(samples) == np.ndarray else np.array(samples)
    #
    def __getitem__(self, item):
        return self._samples[item]
    #
    def __len__(self):
        return len(self._samples)
    #
    def batch_iter(self, batchsize, rand_flg=True):
        indices = np.random.permutation(len(self)) if rand_flg else np.arange(len(self))
        for start in range(0, len(self), batchsize):
            yield self[indices[start: start + batchsize]]
    #
    @classmethod
    def load(cls, data_path, ent_vocab, rel_vocab):
        samples = []
        with open(data_path) as f:
            for line in f:
                sub, rel, obj = line.strip().split('\t')
                samples.append((ent_vocab[sub], rel_vocab[rel], ent_vocab[obj]))
        return TripletDataset(samples)


class Vocab(object):  ##IMP : associate id to each vocab of entities,relations ##todo: not fully mine but modified
    '''
    ##Todo :To use its object :name to indx converter:
    ent_vocab = Vocab.load(args.ent)
    rel_vocab = Vocab.load(args.rel)
    '''
    def __init__(self, vector=None, noheadtail=False, heads=None,tails=None ):
        if noheadtail == False:
            self.ishead = []  ### not binary !!!
            self.istail = []  ### not binary !!!
            if heads is not None:
                self.ishead = heads  ### not binary !!!
            if tails is not None:
                self.istail = tails  ### not binary !!!
        if vector is None:
            self.id2word = []
            self.word2id = {}
            self.noheadtail= noheadtail
        else:
            self.word2id=[]
            self.id2word =vector
            vec=np.asarray(vector)
            for itm in set(vector):
                self.word2id.append(np.where(vec==itm)[0].tolist())

    def add(self, word,  ishead=0, istail=0):
        if word in self.id2word:
            ind=self.id2word.index(word)
            if not self.noheadtail:
                self.ishead[ind]+= ishead
                self.istail[ind]+= istail
        else: # so ind not found
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)
            if not self.noheadtail:
                self.ishead.append(ishead)
                self.istail.append(istail)

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, word):
        return self.word2id[word]

    def save(self,filename):
        with open(filename,'wt') as f:
            if not self.noheadtail:
                for ii,line in enumerate(self.id2word):
                    f.writelines('\t'.join([line, str(self.ishead[ii]), str(self.istail[ii])])+'\n')
            else:
                for ii,line in enumerate(self.id2word):
                    f.writelines(line+'\n')

    @classmethod
    def load(cls, vocab_path):
        v = Vocab()
        with open(vocab_path) as f:
            for i,wrd in enumerate(f):
                if wrd=='' or wrd=='\r':
                    continue
                if i==0:
                    noheadtail= not '\t' in wrd
                    v.noheadtail= noheadtail
                if noheadtail==False:
                    word, ishead, istail = wrd.split('\t')
                    v.add(word.strip(), int(ishead), int(istail))
                else:
                    word= wrd
                    v.add(word.strip())
        return v



def removecols(lil,locs):
    locs.sort(reverse=True)
    for j in locs:
        removecol(lil, j)

def removecol(lil,j):
    from bisect import bisect_left
    if j < 0:
        j += lil.shape[1]
    if j < 0 or j >= lil.shape[1]:
        raise IndexError('column index out of bounds')
    rows = lil.rows
    data = lil.data
    for i in range(lil.shape[0]):
        pos = bisect_left(rows[i], j)
        if pos == len(rows[i]):
            continue
        elif rows[i][pos] == j:
            rows[i].pop(pos)
            data[i].pop(pos)
            if pos == len(rows[i]):
                continue
        for pos2 in range(pos,len(rows[i])):
            rows[i][pos2] -= 1
    lil._shape = (lil._shape[0],lil._shape[1]-1)



