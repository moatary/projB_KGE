import json
import numpy as np
import pandas as pd
import networkx as nx
#from tqdm import tqdm
from scipy import sparse
#from texttable import Texttable
import math
import torch.nn
import torchvision

def read_graph(graph_path):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    :return graph: graph.
    """
    print("\nTarget matrix creation started.\n")
    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist())
    graph.remove_edges_from(graph.selfloop_edges())
    return graph

# def tab_printer(args):
#     """
#     Function to print the logs in a nice tabular format.
#     :param args: Parameters used for the model.
#     """
#     args = vars(args)
#     keys = sorted(args.keys())
#     t = Texttable()
#     t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(), args[k]] for k in keys])
#     print(t.draw())

def feature_calculator(args, graph):  #@# PART OF FEATURES HERE: norder adjacency //other parts: features extracted perRlation perNgrams...
    """
    Calculating the feature tensor.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :return target_matrices: Target tensor.
    """
    index_1 = [edge[0] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()]
    values = [1 for edge in graph.edges()]
    node_count = max(max(index_1)+1,max(index_2)+1)
    adjacency_matrix = sparse.coo_matrix((values, (index_1,index_2)),shape=(node_count,node_count),dtype=np.float32)
    degrees = adjacency_matrix.sum(axis=0)[0].tolist()
    degs = sparse.diags(degrees, [0])
    normalized_adjacency_matrix = degs.dot(adjacency_matrix)
    target_matrices = [normalized_adjacency_matrix.todense()]
    powered_A = normalized_adjacency_matrix
    if args.window_size > 1:
        for power in tqdm(range(args.window_size-1), desc = "Adjacency matrix powers"):
            powered_A = powered_A.dot(normalized_adjacency_matrix)
            to_add = powered_A.todense()
            target_matrices.append(to_add)
    target_matrices = np.array(target_matrices)
    return target_matrices

def adjacency_opposite_calculator(graph):
    """
    Creating no edge indicator matrix.
    :param graph: NetworkX object.
    :return adjacency_matrix_opposite: Indicator matrix.
    """
    adjacency_matrix = sparse.csr_matrix(nx.adjacency_matrix(graph), dtype=np.float32).todense()
    adjacency_matrix_opposite = np.ones(adjacency_matrix.shape) - adjacency_matrix
    return adjacency_matrix_opposite


def xavier_init(size):
    dim = size[1]
    bound = math.sqrt(6) / math.sqrt(2*dim)
    return np.random.uniform(-bound, bound, size=size)


def max_margin(pos_scores, neg_scores): ##todo: not mine
    return np.maximum(0, 1 - (pos_scores - neg_scores))


def sigmoid(x): ##todo: not mine
    return np.tanh(x * 0.5) * 0.5 + 0.5


def softplus(x): ##todo: not mine
    return np.maximum(0, x)+np.log(1+np.exp(-np.abs(-x)))


def circular_convolution(v1, v2): ##todo: not mine
    freq_v1 = np.fft.fft(v1)
    freq_v2 = np.fft.fft(v2)
    return np.fft.ifft(np.multiply(freq_v1, freq_v2)).real


def circular_correlation(v1, v2): ##todo: not mine
    freq_v1 = np.fft.fft(v1)
    freq_v2 = np.fft.fft(v2)
    return np.fft.ifft(np.multiply(freq_v1.conj(), freq_v2)).real



def joinfiles(paths, finalpath):
    str=''
    for filename in paths:
        with open(filename,'rt') as f:
            str+='\n'+f.read()
    with open(finalpath,'wt') as f:
        f.write(str)


def file2list(filename):
    with open(filename,'rt') as f:
        str = f.read()
        if '\r\n' in str:
            str= str.split('\r\n')
        else:
            str=str.split('\n')
    return str


def unique(lst):
    fin=[]
    ind=[]
    cnt=[]
    for num, itm in enumerate(lst):
        if itm not in fin:
            fin.append(itm)
            ind.append(num)
            cnt.append(0)
        else:
            #locate its loc:
            loc= fin.index(itm)
            cnt[loc]+=1
    return fin, ind, cnt


def intersect_mtlb(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]




def generateTunes(*tunes):
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
    listoftunes=[dict(zip(itmkeys,itmvalues)) for itmkeys,itmvalues in zip(keyss,tuness)]

    return listoftunes



