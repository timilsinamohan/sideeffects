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
random.seed(1949)


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



def perform_matrix_reconstruction(bipart_graph):

    model = NMF(n_components=10, init= 'random')
    WW = model.fit_transform(bipart_graph)
    HH = model.components_
    WW = sparse.csr_matrix(WW)
    HH = sparse.csr_matrix(HH)
    preds = WW.dot(HH)

    reconstructed_pub_web_matrix = preds.A

    return reconstructed_pub_web_matrix


def compute_score(TG):

    iteration = 30
    lambda_diff = 1.0
    I = np.eye(n,n,dtype=np.float32)
    V = I + (lambda_diff/iteration) * H
    state_matrix = TG.copy()

    for j in xrange(iteration):
        state_matrix_new = V.dot(state_matrix.T).T
        state_matrix = state_matrix_new.copy()

    return state_matrix


def innerfold(IDX1,IDX2,m,n):
    mask_idx = np.unravel_index(IDX1, (m, n))
    side_effects_drug_relation_copy = matrix.copy()
    target_idx = np.unravel_index(IDX2, (m, n))
    ###making all the links to predict as 0 ###############    
    for i in range(len(mask_idx[0])):
        side_effects_drug_relation_copy[mask_idx[0][i], mask_idx[1][i]] = 0

    side_effects_drug_relation_fact = perform_matrix_reconstruction(side_effects_drug_relation_copy)
    score = compute_score(side_effects_drug_relation_fact)
   
    Ground_Truth = matrix.copy()
    Ground_Truth = np.array(Ground_Truth)
    GTR = Ground_Truth[target_idx]
    ##ranking from high to low scores##
    rank_list = np.argsort(-score[target_idx])
    r = GTR[rank_list]
    k = 5 ##ndcg@5 
    ndcg = ndcg_at_k(r,k, method=1)
    prec, recall, _ = precision_recall_curve(Ground_Truth[target_idx],score[target_idx])
    print "AUC-PR", auc(recall, prec)
    pr_auc = auc(recall, prec)
    fpr, tpr, threshold = metrics.roc_curve(Ground_Truth[target_idx],score[target_idx])
    roc_auc = metrics.auc(fpr, tpr)
    print "AUC-ROC score:",roc_auc
    print "NDCG@5",ndcg

    return roc_auc,pr_auc,ndcg


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
col_names = ["left_side","right_side","similairity"]
df_drugs_sim = pd.read_csv("data/semantic_similarity_side_effects_drugs.txt",sep ="\t",
                 names =col_names, header=None)
source =df_drugs_sim["left_side"]
destination = df_drugs_sim["right_side"]
similarity = df_drugs_sim["similairity"]


###Drugs similarity Network#####
edge_list = zip(source,destination,similarity)
#print edge_list
print "Side effect graph information loading..."
G = nx.Graph()
G.add_weighted_edges_from(edge_list)

###Getting the Bi-Adjacency matrix between side effects and drugs ###################
matrix = biadjacency_matrix(B, row_order= side_effect_nodes, column_order=drug_nodes)
matrix = matrix.A
m = matrix.shape[0]
n = matrix.shape[1]

Drug_Drug_Adj_mat = nx.adjacency_matrix(G, nodelist= drug_nodes,weight='none')

A = np.array(Drug_Drug_Adj_mat.todense(), dtype=np.float64)

weight_matrix = nx.attr_matrix(G, edge_attr='weight', rc_order=drug_nodes)
weight_matrix = np.array(weight_matrix)

heat_matrix = np.zeros([n,n])
#print "Heat Matrix Creation started:"
G = nx.from_numpy_matrix(A)

print "Heat Matrix filling started........"
for i in range(n):
    for j in range(n):
        if A[j,i] == 1.0:
            heat_matrix[i,j] = weight_matrix[j,i]/G.degree(j)
        if (i==j):
            if G.degree(i):
                heat_matrix[i,j] = (-1.0 / G.degree(i)) * sum(weight_matrix[i,:])


print "Heat Matrix completed ............."

H = heat_matrix.copy()

FOLDS = 10
sz = m * n
IDX = list(range(sz))
#fsz = int(sz/FOLDS)
fsz = int(sz /FOLDS)
np.random.shuffle(IDX)
offset = 0
AUC_train_roc = np.zeros(FOLDS)
AUC_train_pr = np.zeros(FOLDS)
ndcg_folds_train = np.zeros(FOLDS)
####for test sets####
AUC_test_roc = np.zeros(FOLDS)
AUC_test_pr = np.zeros(FOLDS)
ndcg_folds = np.zeros(FOLDS)

np.random.shuffle(IDX)
for f in xrange(FOLDS):
    idx_test = IDX[offset:offset + fsz]
    idx_train = np.setdiff1d(IDX, idx_test)
    np.random.shuffle(idx_train)
    print "Fold:",f
    start_time = time.time()
    #IDX1 = random.sample(xrange(sz),fsz)
    idx_train = idx_train[:fsz].tolist()
    print "Train Fold:",f
    AUC_train_roc[f], AUC_train_pr[f],ndcg_folds_train[f] = innerfold(idx_train+idx_test,idx_train,m,n)
    print "Test Fold:",f	
    AUC_test_roc[f],AUC_test_pr[f],ndcg_folds[f] = innerfold(idx_test,idx_test,m,n)
    print("--- %s seconds ---" % (time.time() - start_time)) 
    offset += fsz


print "Mean AUC-PR", AUC_test_pr.mean()," ", "Standard Deviation:", AUC_test_pr.std()
print "Mean AUC-ROC",AUC_test_roc.mean()," ", "Standard Deviation:", AUC_test_roc.std()
print "Mean NDCG:", ndcg_folds.mean(),"  " , " Standard Deviation:", ndcg_folds.std()




