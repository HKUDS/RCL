import numpy as np
import scipy.sparse as sp
import torch

from params import args


def subgraph_construction(data, time_number, user_num, item_num, t_min):
 

    subgraphs = {}  

 

    for t in range(0, time_number):  
        subgraphs[t] = {}

    for t in range(0, time_number):  
        subgraphs[t]['H'] = sp.dok_matrix((item_num, user_num), dtype=np.int)


    data_coo = data.tocoo()
    for i in range(0, len(data_coo.data)):
        tmp_t = (data_coo.data[i] - t_min)/args.time_slot
       
        tmp_t = tmp_t.astype(int)
      
        subgraphs[tmp_t]['H'][data_coo.col[i], data_coo.row[i]] = 1  


    for t in range(0, time_number):
        subgraphs[t]['G'], subgraphs[t]['U'] = generate_G_from_H(subgraphs[t]['H'])
        subgraphs[t]['H'] = None

    return subgraphs 


def generate_G_from_H(H):


    n_edge = H.shape[1]  
    W = sp.diags(np.ones(n_edge)) 
    DV = np.array(H.sum(1))
    DV2 = sp.diags(np.power(DV+1e-8, -0.5).flatten())  
    DE = np.array(H.sum(0))
    DE2 = sp.diags(np.power(DE+1e-8, -0.5).flatten())  
    HT = H.T

    #BHD XP   
    U = DE2 * HT * DV2  
    # U = DE2 * HT   


    #DHWBHD XP
    G = DV2 * H * W * DE2 * HT * DV2  

    
    return matrix_to_tensor(G), matrix_to_tensor(U)


def matrix_to_tensor(cur_matrix):
    if type(cur_matrix) != sp.coo_matrix:
        cur_matrix = cur_matrix.tocoo()  
    indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  
    values = torch.from_numpy(cur_matrix.data) 
    shape = torch.Size(cur_matrix.shape)

    return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda() 