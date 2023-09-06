import numpy as np
from numpy import random
import pickle
from scipy.sparse import csr_matrix
import math
import gc
import time
import random
import datetime
import dgl

import torch as t
import torch.nn as nn
import torch.utils.data as dataloader
import torch.nn.functional as F

import hypergraph_utils
import DataHandler



import model


from params import args
from Utils.TimeLogger import log
import evaluate

if t.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


now_time = datetime.datetime.now()
modelTime = datetime.datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')



class Model():
    def __init__(self):

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int'     
        self.position_file = args.path + args.dataset + '/position_trn_'     

        self.t_max = -1 

        self.t_min = 0x7FFFFFFF
        self.time_number = -1 

        self.user_num = -1
        self.item_num = -1
        self.subgraphs = {}  
        self.behaviors = []
        self.behaviors_data = {}

        #history
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()  #

        #
        self.curEpoch = 0
        # self.isLoadModel = args.isload

        if args.dataset == ('Tmall' or 'Tmall_LH'):
            # self.behaviors = ['buy']
            self.behaviors = ['pv','fav', 'cart', 'buy']  
        elif args.dataset == 'IJCAI_15':
            self.behaviors = ['click','fav', 'cart', 'buy']
            # self.behaviors = ['buy']
        elif args.dataset == 'JD':
            self.behaviors = ['review','browse', 'buy']
            # self.behaviors = ['buy']
        elif args.dataset == 'retailrocket':
            self.behaviors = ['view','cart', 'buy']
            

        self.positional_mat = t.tensor(pickle.load(open(self.position_file+'buy','rb')).todense())
        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:  
                data = pickle.load(fs)
                self.behaviors_data[i] = data  

                if data.get_shape()[0] > self.user_num:  
                    self.user_num = data.get_shape()[0]  
                if data.get_shape()[1] > self.item_num:  
                    self.item_num = data.get_shape()[1]  


                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()

                if self.behaviors[i]==args.target:
                    self.trainMat = data
                    self.trainLabel = 1*(self.trainMat != 0)  
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))  

            tmp_time_number = (self.t_max - self.t_min) / args.time_slot + 1  
            tmp_time_number = tmp_time_number.astype(int)


            if tmp_time_number > self.time_number:
                self.time_number = tmp_time_number  

            print("print time slot: ", self.time_number)
            print("\n")

        self.max_len_dict = self.get_max_len_dict(self.behaviors_data)
        self.max_len = max(self.max_len_dict)
        self.positional_embedding = self.positional_encoding(args.hidden_dim, self.max_len)


        time = datetime.datetime.now()
        print("Build the subgraph:  ", time)

        for i in range(0, len(self.behaviors)):
            beh_subgraphs = hypergraph_utils.subgraph_construction(self.behaviors_data[i], self.time_number, self.user_num, self.item_num, self.t_min)
            self.subgraphs[i] = beh_subgraphs  

        time = datetime.datetime.now()
        print("Build the subgraph:  ", time)
        
        print("user_num: ", self.user_num)
        print("item_num: ", self.item_num)
        print("\n")


        #-------------------------------------------------------------------------------------------------->>>>>

        train_u, train_v = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = DataHandler.RecDataset(train_data, self.item_num, self.trainMat, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)


        # test_data  
        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)

        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        test_data = np.hstack((test_user.reshape(-1,1), test_item.reshape(-1,1))).tolist()
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)  
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)  


        # ------------------------------------------------------------------------------------------------------->>>>>

    def positional_encoding(self, d_model, max_seq_len):  
        
        position_encoding = t.tensor(np.array([
          [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)]))

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = t.zeros([1, d_model])
        position_encoding = t.cat((pad_row, position_encoding))

        position_encoding = nn.Parameter(position_encoding, requires_grad=False).cuda()

        return position_encoding.data

    def get_max_len_dict(self, behaviors_data):
        max_len_dict = []
        for index, value in enumerate(self.behaviors):
            max_len_dict.append(int(max((1*behaviors_data[index]!=0).sum(-1))))
        return max_len_dict

    def prepareModel(self):
        self.modelName = self.getModelName()  #
        self.setRandomSeed()
        self.gnn_layer = eval(args.gnn_layer)  #
        self.hidden_dim = args.hidden_dim

        if args.isload == True:
            self.loadModel(args.loadModelPath)
        else:
            self.model = model.myModel(self.user_num, self.item_num, self.time_number, self.behaviors, self.subgraphs)

        self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)


        if use_cuda:
            self.model = self.model.cuda()

    def innerProduct(self, u, i, j):  
        pred_i = t.sum(t.mul(u,i), dim=1)    
        pred_j = t.sum(t.mul(u,j), dim=1)  
        return pred_i, pred_j

    def innerProduct_positional(self, u, i, j, user, item_i, item_j):  #
        u_p_pos = u + args.positional_rate*F.normalize(self.positional_embedding[self.positional_mat[user, item_i]])
        u_p_neg = u + args.positional_rate*F.normalize(self.positional_embedding[self.positional_mat[user, item_j]])
        i_p = i + args.positional_rate*F.normalize(self.positional_embedding[self.positional_mat[user, item_i]])
        j_p = j + args.positional_rate*F.normalize(self.positional_embedding[self.positional_mat[user, item_j]])
        pred_i = t.sum(t.mul(u_p_pos,i_p), dim=1)    
        pred_j = t.sum(t.mul(u_p_neg,j_p), dim=1)  

        return pred_i, pred_j


    def run(self):


        self.prepareModel()
        if args.isload == True:
            print("----------------------pre test:")
            HR, NDCG = self.testEpoch(self.test_loader)
            print(f"HR: {HR} , NDCG: {NDCG}")    

        log('Model Prepared')


        cvWait = 0  
        self.best_HR = 0 
        self.best_NDCG = 0
        flag = 0

        print("Test before train:")
        HR, NDCG = self.testEpoch(self.test_loader)

        for e in range(self.curEpoch, args.epoch+1):  
            test = (e % args.tstEpoch == 0)  
            self.curEpoch = e
            log("*****************start epoch %d: ************************"%e)  

            if args.isJustTest == False:
                epoch_loss = self.trainEpoch()
                self.train_loss.append(epoch_loss)  
                log("epoch %d/%d, epoch_loss=%.2f"% (e, args.epoch, epoch_loss))
                self.train_loss.append(epoch_loss)
            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            self.scheduler.step()

            if HR > self.best_HR:
                self.saveHistory()
                self.saveModel()
                self.best_HR = HR
                self.best_epoch = self.curEpoch 
                cvWait = 0
                print("--------------------------------best_HR", self.best_HR)
                # print("-----------------------------------NDCG", self.best_NDCG)
            
            
            if NDCG > self.best_NDCG:
                self.saveHistory()
                self.saveModel()
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch 
                cvWait = 0
                # print("-----------------------------------------------HR", self.best_HR)
                print("-----------------------------------------------best_NDCG", self.best_NDCG)
            

            if (HR<self.best_HR) and (NDCG<self.best_NDCG): 
                cvWait += 1


            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                self.saveHistory()
                self.saveModel()
                break


        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)


    def sim(self, z1, z2):
        z1 = F.normalize(z1)  
        z2 = F.normalize(z2)

        return t.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=1024):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: t.exp(x / args.tau)   #        

        indices = t.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))  
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))  

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = t.cat(tmp_refl_sim_list, dim=-1)
            between_sim = t.cat(tmp_between_sim_list, dim=-1)


            losses.append(-t.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()/ (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())+1e-8))

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list
                   
        loss_vec = t.cat(losses)
        return loss_vec.mean()

    def sampleTrainBatch_dgl(self, batIds, pos_id=None, g=None, g_neg=None, sample_num=None, sample_num_neg=None):

        sub_g = dgl.sampling.sample_neighbors(g.cpu(), {'user':batIds}, sample_num, edge_dir='out', replace=True)
        row, col = sub_g.edges()
        row = row.reshape(len(batIds), sample_num)
        col = col.reshape(len(batIds), sample_num)

        if g_neg==None:
            return row, col
        else: 
            sub_g_neg = dgl.sampling.sample_neighbors(g_neg, {'user':batIds}, sample_num_neg, edge_dir='out', replace=True)
            row_neg, col_neg = sub_g_neg.edges()
            row_neg = row_neg.reshape(len(batIds), sample_num_neg)
            col_neg = col_neg.reshape(len(batIds), sample_num_neg)
            return row, col, col_neg 


    def trainEpoch(self):

        train_loader = self.train_loader

        train_loader.dataset.ng_sample()

        epoch_loss = 0
        cnt = 0
        for user, item_i, item_j in train_loader:  

            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()


            user_embed, item_embed, embedding_dict_after_gnn, embedding_dict_after_gnn_dynamic_ssl, user_embedding_before_attention = self.model(self.subgraphs)

            #----ssl-------------------------------------------------------------------------------------------------------------
            #ori
            long_contrastive_loss = 0
            for beh_ in range(len(self.behaviors)):
                long_contrastive_loss += self.batched_contrastive_loss(user_embedding_before_attention[self.behaviors[beh_]][user], user_embed[user])
            #dynamic
            short_contrastive_loss = 0
            for time_ in range(len(embedding_dict_after_gnn)):
                for beh_ in range(len(self.behaviors)):
                    short_contrastive_loss += self.batched_contrastive_loss(embedding_dict_after_gnn[time_][self.behaviors[beh_]][user], embedding_dict_after_gnn_dynamic_ssl[time_][user])
            #----ssl-------------------------------------------------------------------------------------------------------------

            userEmbed = user_embed[user]
            posEmbed = item_embed[item_i]
            negEmbed = item_embed[item_j]

            pred_i, pred_j = self.innerProduct_positional(userEmbed, posEmbed, negEmbed, user, item_i, item_j)

            bprloss = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log().sum()  
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = 0.5 * (bprloss + args.reg * regLoss) / args.batch + args.cl_long_rate*long_contrastive_loss + args.cl_short_rate*short_contrastive_loss 
            epoch_loss = epoch_loss + bprloss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()


            cnt+=1
            log('step %d, step_loss = %f'%(cnt, loss.item()), save=False, oneline=True)  
        log("finish train")
        return epoch_loss

    def testEpoch(self, data_loader, save=False):

        epochHR, epochNDCG = [0]*2
        user_embed, item_embed, embedding_dict_after_gnn, embedding_dict_after_gnn_dynamic_ssl, user_embedding_before_attention = self.model(self. subgraphs)

        cnt = 0
        tot = 0
        for user, item_i in data_loader:
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)  
            userEmbed = user_embed[user_compute]  #
      
            itemEmbed = item_embed[item_compute]
            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)  

            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)  
            epochHR = epochHR + hit  
            epochNDCG = epochNDCG + ndcg  #
            cnt += 1 
            tot += user.shape[0]


        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG


    def sampleTestBatch(self, batch_user_id, batch_item_id):

        batch = len(batch_user_id)
        tmplen = (batch*100)

        sub_trainMat = self.trainMat[batch_user_id].toarray()  
        user_item1 = batch_item_id  
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i]  
            negset = np.reshape(np.argwhere(sub_trainMat[i]==0), [-1])  
            pvec = self.labelP[negset]  
            pvec = pvec / np.sum(pvec)  
            
            random_neg_sam = np.random.permutation(negset)[:99]  
            user_item100_one_user = np.concatenate(( random_neg_sam, np.array([pos_item])))  
            user_item100[i] = user_item100_one_user

            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1

        return user_compute, item_compute, user_item1, user_item100

    def calcRes(self, pred_i, user_item1, user_item100): 

        hit = 0
        ndcg = 0


        for j in range(pred_i.shape[0]):

            _, shoot_index = t.topk(pred_i[j], args.shoot) 
            shoot_index = shoot_index.cpu()
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()

            if type(shoot)!=int and (user_item1[j] in shoot):  
                hit += 1  
                ndcg += np.reciprocal( np.log2( shoot.index( user_item1[j])+2))  
            elif type(shoot)==int and (user_item1[j] == shoot):
                hit += 1  
                ndcg += np.reciprocal( np.log2( 0+2))

        return hit, ndcg  

    def setRandomSeed(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)

    def getModelName(self):  
        title = args.title
        ModelName = \
        args.point + \
        "_" + title + \
        "_" +  args.dataset +\
        "_" + modelTime + \
        "_lr_" + str(args.lr) + \
        "_reg_" + str(args.reg) + \
        "_batch_size_" + str(args.batch) + \
        "_time_slot_" + str(args.time_slot) + \
        "_gnn_layer_" + str(args.gnn_layer)

        return ModelName

    def saveHistory(self):  
        history = dict()
        history['loss'] = self.train_loss  
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        with open(r'./History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs:  
            pickle.dump(history, fs)


    def saveModel(self):  
        ModelName = self.modelName

        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            'model': self.model,
            'history': history,
        }
        t.save(params, savePath)

    def loadModel(self, loadPath):     
        ModelName = self.modelName
        loadPath = loadPath
        checkpoint = t.load(loadPath)
        self.model = checkpoint['model']

        self.curEpoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']



if __name__ == '__main__':

    print(args)
    my_model = Model()  
    my_model.run()

