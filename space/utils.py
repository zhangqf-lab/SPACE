#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 20:52:20 2022

@author: liyuzhe
"""
import os
import scanpy as sc
import numpy as np
import scipy as sci
from scipy.spatial import distance
import networkx as nx
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_scatter import scatter_mean,scatter_sum
from .layer import GAT_Encoder
from .model import SPACE_Graph


def celltype_connectivity(adata,domain_key= 'Label',celltype_key = 'cluster',nboot=100, outdir='./data'):
    import holoviews as hv
    import squidpy as sq
    import pandas as pd
    from tqdm import tqdm
    import random

    random.seed(42)
    os.makedirs(outdir, exist_ok=True)

    sq.gr.spatial_neighbors(adata, n_neighs = 20, coord_type = 'generic') 

    for label in adata.obs[domain_key].unique():
        # CCI
        adata_batch = adata[adata.obs[domain_key] == label].copy()  
        cell_types = adata_batch.obs[celltype_key].unique().tolist()
        sq.gr.interaction_matrix(adata_batch, cluster_key=celltype_key)
        CCI1=adata_batch.uns['{}_interactions'.format(celltype_key)].copy()
        CCI2=CCI1.T.copy()
        CCI=((CCI1+CCI2)/2).round()
        pairwise_connection=pd.DataFrame(CCI,
                                         index=adata_batch.obs[celltype_key].cat.categories,
                                         columns=adata_batch.obs[celltype_key].cat.categories)
        
        pairwise_connection[pairwise_connection<=0]=1e-10
        
        
        df_p=pd.DataFrame(columns=['celltype1','celltype2','p'])
        for i in range(0,len(cell_types)):
            for j in range(i,len(cell_types)):
                df_p=df_p.append(pd.DataFrame({'celltype1':cell_types[i],
                                               'celltype2':cell_types[j],
                                               'p':[0]}),ignore_index=True)
        
        # permutation
        for j in range(nboot):
            adata_p=adata_batch.copy()
            celltype_p=list(adata_batch.obs[celltype_key].values.copy())
            random.shuffle(celltype_p)
            adata_p.obs[celltype_key]=celltype_p
            adata_p.obs[celltype_key]=adata_p.obs[celltype_key].astype('category')
            
            sq.gr.interaction_matrix(adata_p, cluster_key=celltype_key)
            CCI1=adata_p.uns['{}_interactions'.format(celltype_key)].copy()
            CCI2=CCI1.T.copy()
            CCI=((CCI1+CCI2)/2).round()
            pairwise_connection_p = pd.DataFrame(CCI,
                                                 index=adata_p.obs[celltype_key].cat.categories,
                                                 columns=adata_p.obs[celltype_key].cat.categories)
            pairwise_connection_p[pairwise_connection_p<=0]=1e-10
            
            
            pairwise_connection_d=pairwise_connection_p-pairwise_connection
            for i in df_p.index:
                idx1=df_p.loc[i,'celltype1']
                idx2=df_p.loc[i,'celltype2']
                if pairwise_connection_d.loc[idx1,idx2]>=0:
                    df_p.loc[i,'p']=df_p.loc[i,'p']+1
            
        df_p['p']=df_p['p']/nboot
        
        # save data
        celltype_sum=pd.DataFrame(adata_batch.obs[celltype_key].value_counts())
        celltype_sub=celltype_sum[celltype_sum[celltype_key]>=10].copy()
        df=pairwise_connection.loc[celltype_sum.index,celltype_sum.index].copy()
        
        df.to_csv(os.path.join(outdir,'{}_prob.csv'.format(label)))
        celltype_sub.to_csv(os.path.join(outdir,'{}_size.csv'.format(label)))
        
        # from statsmodels.stats.multitest import fdrcorrection
        # df_p['p_adj']=fdrcorrection(df_p['p'])[1]
        
        df_p_m=pd.DataFrame(index=celltype_sum.index,columns=celltype_sum.index)
        for i in df_p.index:
            idx1=df_p.loc[i,'celltype1']
            idx2=df_p.loc[i,'celltype2']
            df_p_m.loc[idx1,idx2]=df_p.loc[i,'p']
            df_p_m.loc[idx2,idx1]=df_p.loc[i,'p']
            # df_p_m.loc[idx1,idx2]=df_p.loc[i,'p_adj']
            # df_p_m.loc[idx2,idx1]=df_p.loc[i,'p_adj']
        df_p_m = df_p_m.fillna(1)
        df_p_m[df_p_m==0]=1e-10
        df_p.to_csv(os.path.join(outdir,'{}_p.csv'.format(label)),index=None)
        df_p_m.to_csv(os.path.join(outdir,'{}_prob_p.csv'.format(label)))
           
    return adata


def mad_gap(x, edge_index):

    #eq 1-2 masked cos similarity
    with torch.no_grad():
        dij = torch.cosine_similarity(x[edge_index[0]], x[edge_index[1]], dim=1, eps=1e-8)
        dij = 1 - dij
        # M^{tgt} = A
        d_bar = scatter_sum(dij, edge_index[1])

    return d_bar.cpu()


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

def load_SPACE_model(adata,k,a,heads,graph_const_method='SPACE',loss_type='MSE',device='cpu',outdir='./'):
    print('Construct Graph')
    if graph_const_method=='SPACE':
        graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0],k=k)
        adj = graph_dict.toarray()
    elif graph_const_method=='Squidpy':
        sq.gr.spatial_neighbors(adata, coord_type="generic",n_neighs=k)
        adj = adata.obsp['spatial_connectivities'].toarray()
    print('Average links: {:.2f}'.format(np.sum(adj>0)/adj.shape[0]))
    
    G = nx.from_numpy_array(adj).to_undirected() 
    edge_index = (torch_geometric.utils.convert.from_networkx(G)).edge_index
    
    if sci.sparse.issparse(adata.X):
        X_hvg = adata.X.toarray()
    else:
        X_hvg = adata.X.copy()
    
    if loss_type == 'BCE':
        scaler = MaxAbsScaler() # MinMaxScaler() 
        scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg))
    elif loss_type == 'MSE':
        scaled_x = torch.from_numpy(X_hvg) 
    
    #prepare training data
    data_obj = Data(edge_index=edge_index, x=scaled_x) 
    data_obj.num_nodes = X_hvg.shape[0] 
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
    transform = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, 
                                  add_negative_train_samples=False, split_labels=True)
    train_data, _ , _ = transform(data_obj) 
    num_features = data_obj.num_features
    print('Load SPACE model')
    encoder = GAT_Encoder(
        in_channels=num_features,
        num_heads={'first':heads,'second':heads,'mean':heads},
        hidden_dims=[128,128],
        dropout=[0.3,0.3],
        concat={'first': True, 'second': True})
    model = SPACE_Graph(encoder= encoder,decoder=None,loss_type=loss_type)
    print(model) 
    model.to(device)
    
    # Load model
    pretrained_dict = torch.load(os.path.join(outdir,'model.pt'), map_location=device)                            
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    model = model.eval()

    x, edge_index = data_obj.x.to(torch.float).to(device), data_obj.edge_index.to(torch.long).to(device)
    z_nodes, attn_w = model.encode(x, edge_index)
    
    graph_loss = model.graph_loss(z_nodes, train_data.pos_edge_label_index) 
    reconstructed_features = model.decoder_x(z_nodes)
    
    if loss_type=='BCE': 
        feature_loss = torch.nn.functional.binary_cross_entropy(reconstructed_features, x) 
    elif loss_type=='MSE':
        feature_loss = torch.nn.functional.mse_loss(reconstructed_features, x) 
    
    return feature_loss.cpu().detach().numpy(), graph_loss.cpu().detach().numpy()

def load_SPACE_model_full(adata,k,a,heads,graph_const_method='SPACE',loss_type='MSE',device='cpu',outdir='./'):
    print('Construct Graph')
    if graph_const_method=='SPACE':
        graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0],k=k)
        adj = graph_dict.toarray()
    elif graph_const_method=='Squidpy':
        sq.gr.spatial_neighbors(adata, coord_type="generic",n_neighs=k)
        adj = adata.obsp['spatial_connectivities'].toarray()
    print('Average links: {:.2f}'.format(np.sum(adj>0)/adj.shape[0]))
    
    G = nx.from_numpy_array(adj).to_undirected() 
    edge_index = (torch_geometric.utils.convert.from_networkx(G)).edge_index
    
    if sci.sparse.issparse(adata.X):
        X_hvg = adata.X.toarray()
    else:
        X_hvg = adata.X.copy()
    
    if loss_type == 'BCE':
        scaler = MaxAbsScaler() # MinMaxScaler() 
        scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg))
    elif loss_type == 'MSE':
        scaled_x = torch.from_numpy(X_hvg) 
    
    #prepare training data
    data_obj = Data(edge_index=edge_index, x=scaled_x) 
    data_obj.num_nodes = X_hvg.shape[0] 
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
    transform = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, 
                                  add_negative_train_samples=False, split_labels=True)
    train_data, _ , _ = transform(data_obj) 
    num_features = data_obj.num_features
    print('Load SPACE model')
    encoder = GAT_Encoder(
        in_channels=num_features,
        num_heads={'first':heads,'second':heads,'mean':heads},
        hidden_dims=[128,128],
        dropout=[0.3,0.3],
        concat={'first': True, 'second': True})
    model = SPACE_Graph(encoder= encoder,decoder=None,loss_type=loss_type)
    print(model) 
    model.to(device)
    
    # Load model
    pretrained_dict = torch.load(os.path.join(outdir,'model.pt'), map_location=device)                            
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    model = model.eval()

    x, edge_index = data_obj.x.to(torch.float).to(device), data_obj.edge_index.to(torch.long).to(device)
    z_nodes, attn_w = model.encode(x, edge_index)
    
    graph_loss = model.graph_loss(z_nodes, train_data.pos_edge_label_index) 
    reconstructed_features = model.decoder_x(z_nodes)
    
    if loss_type=='BCE': 
        feature_loss = torch.nn.functional.binary_cross_entropy(reconstructed_features, x,reduction='none') 
    elif loss_type=='MSE':
        feature_loss = torch.nn.functional.mse_loss(reconstructed_features, x,reduction='none') 
    
    return feature_loss.cpu().detach().numpy(), graph_loss.cpu().detach().numpy()


def extract_attention(adata,k,a,heads,graph_const_method='SPACE',loss_type='MSE',device='cpu',outdir='./',save_attn=False):
    print('Construct Graph')
    if graph_const_method=='SPACE':
        graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0],k=k)
        adj = graph_dict.toarray()
    elif graph_const_method=='Squidpy':
        sq.gr.spatial_neighbors(adata, coord_type="generic",n_neighs=k)
        adj = adata.obsp['spatial_connectivities'].toarray()
    print('Average links: {:.2f}'.format(np.sum(adj>0)/adj.shape[0]))
    
    G = nx.from_numpy_array(adj).to_undirected() 
    edge_index = (torch_geometric.utils.convert.from_networkx(G)).edge_index
    
    if sci.sparse.issparse(adata.X):
        X_hvg = adata.X.toarray()
    else:
        X_hvg = adata.X.copy()
    
    if loss_type == 'BCE':
        scaler = MaxAbsScaler() # MinMaxScaler() 
        scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg))
    elif loss_type == 'MSE':
        scaled_x = torch.from_numpy(X_hvg) 
    
    #prepare training data
    data_obj = Data(edge_index=edge_index, x=scaled_x) 
    data_obj.num_nodes = X_hvg.shape[0] 
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None
    transform = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, 
                                  add_negative_train_samples=False, split_labels=True)
    train_data, _ , _ = transform(data_obj) 
    num_features = data_obj.num_features
    print('Load SPACE model')
    encoder = GAT_Encoder(
        in_channels=num_features,
        num_heads={'first':heads,'second':heads,'mean':heads},
        hidden_dims=[128,128],
        dropout=[0.3,0.3],
        concat={'first': True, 'second': True})
    model = SPACE_Graph(encoder= encoder,decoder=None,loss_type=loss_type)
    print(model) 
    model.to(device)
    
    # Load model
    pretrained_dict = torch.load(os.path.join(outdir,'model.pt'), map_location=device)                            
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    model = model.eval()

    x, edge_index = data_obj.x.to(torch.float).to(device), data_obj.edge_index.to(torch.long).to(device)
    z_nodes, attn_w = model.encode(x, edge_index)
    
    if save_attn:
        for i in range(2):
            edge_index, attention_weights = attn_w[i]
            edge_index, attention_weights = edge_index.detach().cpu(), attention_weights.detach().cpu()
            
            layer_filename = f'GAT_Layer_{i+1}_edge_index.pt'
            edges_filepath = os.path.join(outdir, layer_filename)
            layer_filename = f'GAT_Layer_{i+1}_attention_weights.pt'
            attn_w_filepath = os.path.join(outdir, layer_filename)
            
            torch.save(edge_index, edges_filepath)
            torch.save(attention_weights, attn_w_filepath)
    
    return attn_w


def extract_attn_data(edge_index_attn, weights_attn, dim=None):
    edges = edge_index_attn.T
    if not dim:
        w = weights_attn.mean(dim=1)
    else:
        w = weights_attn[:, dim]
    w = w.squeeze()
    top_values, top_indices = torch.topk(w, len(edges))
    top_edges = edges[top_indices]
    
    return top_edges, top_values
