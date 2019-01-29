#!/usr/bin/env python
import pandas as pd
import networkx as nx
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import normalize
import random
from sklearn import metrics
import time
from sklearn.decomposition import NMF
import random
from scipy import sparse
np.random.seed(1957)

def dcg_at_k(r, k, method=0):
   
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):

    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def perform_matrix_reconstruction(get_pub_web_matrix):

    model = NMF(n_components=10, init= 'random')
    W = model.fit_transform(get_pub_web_matrix)
    H = model.components_
    #print "Matrix factorization completed"

    W = sparse.csr_matrix(W)
    H = sparse.csr_matrix(H)
    preds = W.dot(H)

    reconstructed_pub_web_matrix = preds.A

    return reconstructed_pub_web_matrix


def compute_score(TG):

    iteration = 3
    lambda_diff = 0.5
    I = np.eye(n,n,dtype=np.float32)
    V = I + (lambda_diff/iteration) * H
    state_matrix = TG.copy()

    for j in xrange(iteration):
        state_matrix_new = V.dot(state_matrix.T).T
        state_matrix = state_matrix_new.copy()

    return state_matrix


def innerfold(IDX,m,n):
    mask_idx = np.unravel_index(IDX, (m, n))
    side_effects_drug_relation_copy = matrix.copy()
    target_idx = np.unravel_index(IDX, (m, n))
    
    for i in range(len(mask_idx[0])):
        side_effects_drug_relation_copy[mask_idx[0][i], mask_idx[1][i]] = 0

    side_effects_drug_relation_fact = perform_matrix_reconstruction(side_effects_drug_relation_copy)
    
    ###Before starrting diffusion lets use only 20% of the labelled nodes for ranking ####
    get_random_index = np.random.choice(range(n), size=int(n*0.5), replace=False)
    side_effects_drug_relation_fact[idx_query,:][get_random_index]=0

    score = compute_score(side_effects_drug_relation_fact)
    
    ####Retrieve the score for the query and rank this #######################################
    rank_list = np.argsort(-score[idx_query,get_random_index])
    #print "Ground truth for this:,",GR_TR[rank_list]
    r = GR_TR[rank_list]
    k = 5
    results = ndcg_at_k(r,k, method=1)
    print "NDCG:",results
    

    return results

df = pd.read_csv("data/side-effect-and-drug_name.tsv",sep = "\t")
drug_id = df["drugbank_id"]
drug_name = df["drugbank_name"]
side_effect =df["side_effect_name"]
edgelist1 = zip(side_effect, drug_name)

##making Biparite Graph##
B = nx.DiGraph()
B.add_nodes_from(side_effect,bipartite = 0)
B.add_nodes_from(drug_name,bipartite = 1)
B.add_edges_from(edgelist1)

drug_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==1}
side_effect_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==0}
side_effect_nodes = list(side_effect_nodes)

col_names = ["left_side","right_side","similairity"]
df_drugs_sim = pd.read_csv("data/semantic_similarity_side_effects_drugs.txt",sep ="\t",
                 names =col_names, header=None)



source =df_drugs_sim["left_side"]
destination = df_drugs_sim["right_side"]
similarity = df_drugs_sim["similairity"]


###Drugs similarity Network#####
edge_list = zip(source,destination,similarity)
#print edge_list
print "Side effect graph information loading....."
G = nx.Graph()
G.add_weighted_edges_from(edge_list)

matrix = biadjacency_matrix(B, row_order= side_effect_nodes, column_order=drug_nodes)
matrix = matrix.A
m = matrix.shape[0]
n = matrix.shape[1]

#query = "Gastric ulcer"
#query = "Angioedema"
#query = "Suicide"
#query = "Nausea"
#query = "Diarrhoea"
#query = "Constipation"
query = "Anaemia"
#query = "Anaemia megaloblastic"
idx_query = side_effect_nodes.index(query)
GR_TR = matrix[idx_query,:] 
#print idx
#print "Ground Truth:", len(matrix[idx_query,:])

Drug_Drug_Adj_mat = nx.adjacency_matrix(G, nodelist= drug_nodes,weight='none')

A = np.array(Drug_Drug_Adj_mat.todense(), dtype=np.float64)

weight_matrix = nx.attr_matrix(G, edge_attr='weight', rc_order=drug_nodes)
weight_matrix = np.array(weight_matrix)

heat_matrix = np.zeros([n,n])
#print "Heat Matrix Creation started:"
G = nx.from_numpy_matrix(A)

print "Heat Matrix filling started:"
for i in range(n):
    for j in range(n):
        if A[j,i] == 1.0:
            heat_matrix[i,j] = weight_matrix[j,i]/G.degree(j)
        if (i==j):
            if G.degree(i):
                heat_matrix[i,j] = (-1.0 / G.degree(i)) * sum(weight_matrix[i,:])


print "Heat Matrix Completed:"

H = heat_matrix.copy()

sz = m * n
IDX = list(range(sz))
#fsz = int(sz/FOLDS)
fsz = int(sz * 0.80)
print "Total number of test sets:",fsz
print "Total number of sets:", sz
np.random.shuffle(IDX)
offset = 0
start_time = time.time()
IDX1 = random.sample(xrange(sz),fsz)
innerfold(IDX1,m,n)

print("--- %s seconds ---" % (time.time() - start_time))




