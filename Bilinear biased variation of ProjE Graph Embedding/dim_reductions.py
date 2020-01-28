import numpy as np
from scipy import sparse


def missingvalueratio(input,finaldim):
    zerocount=np.asarray(input.sum(0))[0]
    inds=np.argsort(zerocount)
    transform=np.zeros((input.shape[1],finaldim))
    inds2set=inds[-finaldim:]
    transform[inds2set,list(range(len(inds2set)))]=1
    return transform,input[:,inds2set]


def pca(input,finaldim):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=finaldim)
    res=pca.fit_transform(input.todense())
    return pca.components_.transpose(), res


def svd(input,finaldim):
    # from sklearn.decomposition import TruncatedSVD
    # svd=TruncatedSVD(n_components=finaldim, random_state=42)
    # res=svd.fit_transform(input.todense())
    from scipy.sparse.linalg import svds
    u, s, v = svds(input, k=finaldim)
    X = u.dot(np.diag(s))  # output of TruncatedSVD
    return v.transpose(),X


def tsne(input,finaldim):
    from sklearn.manifold import TSNE
    return [],TSNE(n_components=finaldim, n_iter=300).fit_transform(input.todense())


def fastica(input,finaldim):
    from sklearn.decomposition import FastICA
    ICA = FastICA(n_components=finaldim, random_state=12)
    RES=ICA.fit_transform(input.todense())
    return ICA.components_.transpose(), RES


def umap(input,finaldim):
    import umap
    um=umap.UMAP(n_neighbors=max(20,int(input.shape[0]/100)), min_dist=0.3, n_components=finaldim)
    res=um.fit_transform(input.todense())
    return [],res


# def randomforest(input,finaldim):
#     from sklearn.ensemble import RandomForestRegressor
#     model = RandomForestRegressor(random_state=1, max_depth=finaldim)
#     model.fit(input.todense(), train.Item_Outlet_Sales)



def factoranal(input,finaldim):
    input1=input.todense()
    from sklearn.decomposition import FactorAnalysis
    fa=FactorAnalysis(n_components=finaldim)
    res=fa.fit_transform(input1)
    return fa.components_.transpose(), res


def isomap(input,finaldim):
    from sklearn import manifold
    im= manifold.Isomap(n_neighbors=max(20,int(input.shape[0]/100)), n_components=finaldim, n_jobs=-1)
    trans=im.fit_transform(input.todense().transpose())
    return trans,input.todense()*trans



def spectralembed(input,finaldim):
    from sklearn.manifold import SpectralEmbedding
     # - 'nearest_neighbors' : construct affinity matrix by knn graph
     # - 'rbf' : construct affinity matrix by rbf kernel
     # - 'precomputed' : interpret X as precomputed affinity matrix
    embedding = SpectralEmbedding(n_components=finaldim,affinity='rbf')
    X_transformed = embedding.fit_transform(input.todense().transpose())
    return X_transformed,input.todense()*X_transformed



def lowvarfilter(input,finaldim):
    input1=input.todense()
    inputvar=np.asarray(np.var(input1,0))[0]
    inds=np.argsort(inputvar)
    transform=np.zeros((input.shape[1],finaldim))
    inds2set=inds[-finaldim:]
    transform[inds2set,list(range(len(inds2set)))]=1
    return transform,input[:,inds2set]






def initialize_DimRed_Entityembeds_Relationembeds(relembpath='databases/FB15k/weighted#relations75x20features2.pkl', entembpath='databases/FB15k/weighted_entt_1500features.pkl',reldim=100,entdim=100, reltype=['pca'],enttype=['pca']):
    import pickle
    import sklearn
    try:
        import dim_reductions
    except:
        pass
    methodnames=['svd','missingvalueratio','pca','tsne','fastica','factoranal','isomap','lowvarfilter','umap','spectralembed']
    functionnames=[svd,missingvalueratio,pca,tsne,fastica,factoranal,isomap,lowvarfilter,umap,spectralembed]
    outputstate=[]
    if not entembpath is '':
        transforms_ent=[[] for _ in range(len(enttype))]
        with open(entembpath, 'rb') as w:
            ents = pickle.load(w)
            for i,typ in enumerate(enttype):
                print('training%dth:%s'%(i,typ))
                ind=methodnames.index(typ)
                transforms_ent[i]=functionnames[ind](ents,entdim)
                print('finished %dth:%s\n' % (i, typ))
        outputstate.append(transforms_ent)
    if not relembpath is '':
        transforms_rel = [[] for _ in range(len(reltype))]
        with open(relembpath, 'rb') as w:
            rels = pickle.load(w)
            for i, typ in enumerate(reltype):
                print('training%dth:%s' % (i, typ))
                ind = methodnames.index(typ)
                transforms_rel[i] = functionnames[ind](rels, reldim)
                print('finished %dth:%s' % (i, typ))
        outputstate.append(transforms_rel)
    return outputstate



def save_dimreductions(weighted_transform_glossaries,typee,redtype,path='../data/'):
    import os
    import pickle
    try:
        os.mkdir(path+typee)
    except:
        pass
    patt=path+typee
    for ii,nam in enumerate(redtype):
        with open(patt+'/'+nam+'_ltrans.pkl','wb') as ff:
            pickle.dump(weighted_transform_glossaries[ii][0],ff)
        with open(patt+'/'+nam+'_result.pkl','wb') as ff:
            pickle.dump(weighted_transform_glossaries[ii][1],ff)



def load_dimreduction(name='pca',type='entities',database='FB15k',path='../data/',defaultfiletype='_ltrans'):
    import pickle
    with open(path+'dimreds_'+type+'_'+name+defaultfiletype+'.pkl','rb') as ff:
        mtxx= pickle.load(ff)
    return mtxx