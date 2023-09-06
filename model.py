import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from params import args


class myModel(nn.Module):
    def __init__(self, userNum, itemNum, time_number, behavior, subgraphs):  
        super(myModel, self).__init__()  

        self.userNum = userNum
        self.itemNum = itemNum
        self.time_number = time_number
        self.behavior = behavior
        self.subgraphs = subgraphs
        self.embedding_dict = self.init_embedding() 
        self.weight_dict, self.time_attention_weight_dict = self.init_weight()
        self.hgnns = self.init_hgnns() 
        self.act = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(args.drop_rate)
        self.layer_norm = torch.nn.LayerNorm(args.hidden_dim)  
        self.self_attention_net = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim*2),
            nn.Dropout(args.drop_rate),
            nn.PReLU(),
            nn.Linear(args.hidden_dim*2, args.hidden_dim),
            nn.Dropout(args.drop_rate),
            nn.PReLU()
        )


    def init_embedding(self):
 
        times_user_embedding = {}
        times_item_embedding = {}
        for t in range(0, self.time_number):
            times_user_embedding[t] = {}
            times_item_embedding[t] = {}
  
        embedding_dict = {  
            'times_user_embedding': times_user_embedding,
            'times_item_embedding': times_item_embedding,
        }

        return embedding_dict

    def init_weight(self):  

        initializer = nn.init.xavier_uniform_

        weight_dict = nn.ParameterDict({
            'w_q': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_k': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_v': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_d_d': nn.Parameter(initializer(torch.empty([args.hidden_dim, 1]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.hidden_dim, args.hidden_dim]))),
            'alpha': nn.Parameter(torch.ones(2)),
        })

        time_attention_weight_dict = nn.ParameterList()
        for t in range(0, self.time_number):
            time_attention_weight_dict.append(nn.Parameter(initializer(torch.empty([args.hidden_dim, 1]))))

        return weight_dict, time_attention_weight_dict

    def init_hgnns(self):
        hgnns = nn.ModuleList()
        for t in range(0, self.time_number):
            hgnns.append(nn.ModuleDict())
            for beh in self.behavior:
                hgnns[t][beh] = HGNN(self.userNum, self.itemNum)

        weights = hgnns[0][self.behavior[0]].state_dict()
        for t in range(0, self.time_number):
            for beh in self.behavior:
                hgnns[t][beh].load_state_dict(weights)


        return hgnns

    def init_attention(self):
        pass

    def forward(self, subgraphs):

        embedding_dict_after_gnn_dynamic_ssl =  {}

        for t in range(0, self.time_number):
            for i, beh in enumerate(self.behavior):
                model = self.hgnns[t][beh]
                if t == 0:
                    self.embedding_dict['times_item_embedding'][t][beh], self.embedding_dict['times_user_embedding'][t][beh] = model(self.subgraphs[i][t]['G'], self.subgraphs[i][t]['U'] , model.item_embedding.weight, model.user_embedding.weight)  
                else:
                    self.embedding_dict['times_item_embedding'][t][beh], self.embedding_dict['times_user_embedding'][t][beh] = model(self.subgraphs[i][t]['G'], self.subgraphs[i][t]['U'], self.embedding_dict['times_item_embedding'][t-1][beh], self.embedding_dict['times_user_embedding'][t-1][beh])

            embedding_dict_after_gnn_dynamic_ssl[t] = self.time_behavior_attention(self.embedding_dict['times_user_embedding'][t], t)


        embedding_dict_after_gnn = self.embedding_dict['times_user_embedding']


        for t in range(0, self.time_number):
            if t==0:
                continue
            else:  
                
                user_z = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['times_user_embedding'][t-1], self.embedding_dict['times_user_embedding'][t])
                item_z = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['times_item_embedding'][t-1], self.embedding_dict['times_item_embedding'][t])
                
                for i, beh in enumerate(self.behavior):
                    self.embedding_dict['times_user_embedding'][t][beh] = (self.embedding_dict['times_user_embedding'][t][beh] + user_z[i]) /2
                    self.embedding_dict['times_item_embedding'][t][beh] = (self.embedding_dict['times_item_embedding'][t][beh] + item_z[i]) /2

        user_embedding_before_attention = self.embedding_dict['times_user_embedding'][self.time_number-1]
        item_embedding_before_attention = self.embedding_dict['times_item_embedding'][self.time_number-1]

        user_embedding = self.behavior_attention(self.embedding_dict['times_user_embedding'][self.time_number-1])
        item_embedding = self.behavior_attention(self.embedding_dict['times_item_embedding'][self.time_number-1])
        
        return user_embedding, item_embedding, embedding_dict_after_gnn, embedding_dict_after_gnn_dynamic_ssl, user_embedding_before_attention


    def para_dict_to_tenser(self, para_dict):  
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])

        tensors = torch.stack(tensors, dim=0)

        return tensors

    def self_attention(self, trans_w, embedding_t_1, embedding_t):  
        """
        """

        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], args.hidden_dim/args.head_num


        Q = torch.matmul(q, trans_w['w_q'])  
        K = torch.matmul(k, trans_w['w_k'])
        V = torch.matmul(v, trans_w['w_v'])

        Q = torch.unsqueeze(Q, 1)  
        K = torch.unsqueeze(K, 0)  
        V = torch.unsqueeze(V, 0)  

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))
        att = torch.sum(att, dim=-1) 
        att = torch.unsqueeze(att, dim=-1)  
        att = F.softmax(att, dim=1)  
        self.self_attention_para = nn.Parameter(att)

        Z = torch.mul(att, V)  
        Z = torch.sum(Z, dim=1)  


        return Z

    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):  
       
        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], args.hidden_dim/args.head_num

        Q = torch.matmul(q, trans_w['w_q']) 
        K = torch.matmul(k, trans_w['w_k'])
        V = torch.matmul(v, trans_w['w_v'])  


        Q = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)  
        K = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)

        Q = torch.unsqueeze(Q, 2)  
        K = torch.unsqueeze(K, 1)  
        V = torch.unsqueeze(V, 1)  

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  
        att = torch.sum(att, dim=-1) 
        att = torch.unsqueeze(att, dim=-1)  
        att = F.softmax(att, dim=2)  

        self.multi_head_self_attention_para = nn.Parameter(att)

        Z = torch.mul(att, V)  
        Z = torch.sum(Z, dim=2)  

        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])
      
        return Z

    def time_behavior_attention(self, embedding_input, t): 
        embedding = self.para_dict_to_tenser(embedding_input)  
        attention = torch.matmul(embedding, self.time_attention_weight_dict[t])  
        attention = F.softmax(attention, dim=0)*2.5  
        self.attention_para = nn.Parameter(attention)

        Z = torch.mul(attention, embedding)  
        Z = torch.sum(Z, dim=0)  

        return Z

    def behavior_attention(self, embedding_input):  
        embedding = self.para_dict_to_tenser(embedding_input)  
        attention = torch.matmul(embedding, self.weight_dict['w_d_d'])  
        attention = F.softmax(attention, dim=0)*2.5  
        self.attention_para = nn.Parameter(attention)

        Z = torch.mul(attention, embedding)  #[beh, N, 1][beh, N, d]==>[beh, N, d]
        Z = torch.sum(Z, dim=0)  #[beh, N, d]==>[N, d]

        return Z


class HGNN(nn.Module):
    def __init__(self, userNum, itemNum):
        super(HGNN, self).__init__()  
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim
        self.sigmoid = torch.nn.Sigmoid()
        self.user_embedding, self.item_embedding = self.init_embedding()
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)
        self.gnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.gnn_layer)):  
            self.layers.append(HGNNLayer(args.hidden_dim, args.hidden_dim, weight=True, activation=self.act))  

    def init_embedding(self):

        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim)
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)


        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))  
        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim))
        i_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        u_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(i_concatenation_w)
        init.xavier_uniform_(u_concatenation_w)
        init.xavier_uniform_(i_input_w)
        init.xavier_uniform_(u_input_w)

        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w


    def forward(self, G, U, input_item_embedding, input_user_embedding):
        all_item_embeddings = []
        all_user_embeddings = []

        #--------------------------------alpha---------------------------------------------------------
        self.alpha.data = self.sigmoid(self.alpha)  
        item_embedding = self.alpha[0]*input_item_embedding + (1-self.alpha[0])*self.item_embedding.weight
        user_embedding = self.alpha[1]*input_user_embedding + (1-self.alpha[1])*self.user_embedding.weight

        item_embedding = torch.matmul(item_embedding , self.i_input_w)
        user_embedding = torch.matmul(user_embedding , self.u_input_w)
        #--------------------------------alpha---------------------------------------------------------
        

        for i, layer in enumerate(self.layers):
            item_embedding, user_embedding = layer(G, U, item_embedding, user_embedding)

            norm_item_embeddings = F.normalize(item_embedding, p=2, dim=1)  #TODO:
            norm_user_embeddings = F.normalize(user_embedding, p=2, dim=1)

            all_item_embeddings.append(norm_item_embeddings)
            all_user_embeddings.append(norm_user_embeddings)

        item_embedding = torch.cat(all_item_embeddings, dim=1)
        user_embedding = torch.cat(all_user_embeddings, dim=1)

        item_embedding = torch.matmul(item_embedding , self.i_concatenation_w)
        user_embedding = torch.matmul(user_embedding , self.u_concatenation_w)

        return item_embedding, user_embedding

class HGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, weight=True, activation=None):
        super(HGNNLayer, self).__init__()

        self.act = torch.nn.PReLU()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)

    def forward(self, G, U, item_embedding_para, user_embedding_para):

        item_embedding = torch.mm(G, item_embedding_para)
        item_embedding = torch.mm(item_embedding, self.i_w)
        item_embedding = self.act(item_embedding)

        user_embedding = torch.mm(U, item_embedding)  
        user_embedding = torch.mm(user_embedding, self.u_w)

        user_embedding = self.act(user_embedding)  

        return item_embedding, user_embedding
