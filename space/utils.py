#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 20:52:20 2022

@author: liyuzhe
"""
import scanpy as sc
import numpy as np
from scipy.spatial import distance
import networkx as nx


# Construct adj graph of image-based data 
def graph_computing(adj_coo, cell_num,k):
    edgeList = []
    for node_idx in range(cell_num):
        tmp = adj_coo[node_idx, :].reshape(1, -1)
        distMat = distance.cdist(tmp, adj_coo, 'euclidean')
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0, res[0][1:k+1]]
        boundary = np.mean(tmpdist) + np.std(tmpdist)
        for j in np.arange(1, k+1):
            if distMat[0, res[0][j]] <= boundary:
                weight = 1.0
            else:
                weight = 0.0
            edgeList.append((node_idx, res[0][j], weight))

    return edgeList


def edgeList2edgeDict(edgeList, nodesize):
    graphdict = {}
    tdict = {}
    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1] = ""
        tdict[end2] = ""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1] = tmplist

    # check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i] = []

    return graphdict


def graph_construction(adj_coo, cell_N, k):
    adata_Adj = graph_computing(adj_coo, cell_N, k)
    graphdict = edgeList2edgeDict(adata_Adj, cell_N)
    adj_org = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    
    return adj_org

