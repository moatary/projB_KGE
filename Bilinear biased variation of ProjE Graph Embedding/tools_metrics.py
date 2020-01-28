import math

def nests_weights(i_s, totnums): # weights for each nest of relations:
    reinforce=0.4
    weights=[]
    for i,totnum in zip(i_s, totnums):
        weights.append( 0.01*math.floor(100*( 2/(1+math.exp(-reinforce*(totnum-i+1)))-1) /( 2/(1+math.exp(-reinforce*(totnum+1)))-1)) ) # letting be tanh
    return weights

def nest_weight(i,totnum): # weights for each nest of relations:
    reinforce=0.4
    weights=[]
    return ( 0.01*math.floor(100*( 2/(1+math.exp(-reinforce*(totnum-i+1)))-1) /( 2/(1+math.exp(-reinforce*(totnum+1)))-1)) ) # letting be tanh  # letting be tanh
