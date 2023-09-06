import argparse


def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')

	# #Tmall------------------------------------------------------------------------------------------------------------------------------
	# #for this model
	# parser.add_argument('--hidden_dim', default=16, type=int, help='embedding size')  #
	# parser.add_argument('--gnn_layer', default="[16,16,16,16]", type=str, help='gnn layers: number + dim')  #
	# parser.add_argument('--time_slot', default=864000, type=float, help='length of time slots')  #
	# parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention')  #
	# parser.add_argument('--gate_rate', default=0.8, type=float, help='gating rate')  #
	# parser.add_argument('--point', default='ICDE_CIKM_WOCL', type=str, help='')
	# parser.add_argument('--title', default='self_attention_behavior', type=str, help='title of model')

	# #for train
	# parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  #

	# parser.add_argument('--opt_base_lr', default=1e-4, type=float, help='learning rate')
	# parser.add_argument('--opt_max_lr', default=1e-3, type=float, help='learning rate')
	# parser.add_argument('--opt_weight_decay', default=1e-4, type=float, help='weight decay regularizer')

	# parser.add_argument('--batch', default=4096, type=int, help='batch size')  #
	# parser.add_argument('--reg', default=1.45e-2, type=float, help='weight decay regularizer')  #
	# parser.add_argument('--epoch', default=1000, type=int, help='number of epochs')  #
	# parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')  #
	# parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	# parser.add_argument('--mult', default=1, type=float, help='multiplier for the result')  #
	# parser.add_argument('--drop_rate', default=0.1, type=float, help='drop_rate')  #
	# parser.add_argument('--seed', type=int, default=19)  #
	# parser.add_argument('--slope', type=float, default=0.1)  #
	# parser.add_argument('--patience', type=int, default=300)
	# parser.add_argument('--cl_long_rate', default=0.013, type=float, help='cl_rate')
	# parser.add_argument('--cl_short_rate', default=0.0005, type=float, help='cl_rate')

	# #for save and read
	# parser.add_argument('--path', default='/home/ww/Code/MultiBehavior_BASELINE/MB-GCN/Datasets/', type=str, help='data path')
	# parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	# parser.add_argument('--load_model', default=None, help='model name to load')
	# parser.add_argument('--dataset', default='Tmall', type=str, help='name of dataset')  #
	# parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	# parser.add_argument('--isload', default=False , type=bool, help='whether load model')  #
	# parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')

	# parser.add_argument('--loadModelPath', default='/home/ww/Code/work1/master_behavior_attention/Model/IJCAI_15/ICDE_CIKM_WOCL_self_attention_behavior_IJCAI_15_2022_04_26__11_30_42_lr_0.001_reg_0.0145_batch_size_4096_time_slot_31104000.0_gnn_layer_[16,16].pth', type=str, help='loadModelPath')


	# #use less
	# # parser.add_argument('--memosize', default=2, type=int, help='memory size')
	# parser.add_argument('--sampNum', default=10, type=int, help='batch size for sampling')  #
	# # parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')  #
	# # parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	# parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')  #
	# parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')  #
	# parser.add_argument('--iiweight', default=0.3, type=float, help='weight for ii')  #
	# parser.add_argument('--graphSampleN', default=10000, type=int, help='use 25000 for training and 200000 for testing, empirically')  #
	# parser.add_argument('--divSize', default=1000, type=int, help='div size for smallTestEpoch')
	# parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	# parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
	# parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')  #
	# parser.add_argument('--slot', default=0.5, type=float, help='length of time slots')  #
	# parser.add_argument('--tau', default=0.5, type=float, help='')  #
	# parser.add_argument('--positional_rate', default=0.0001, type=float, help='')  #
	# #Tmall------------------------------------------------------------------------------------------------------------------------------
	


	#IJCAI_15------------------------------------------------------------------------------------------------------------------------------
	#for this model
	parser.add_argument('--hidden_dim', default=16, type=int, help='embedding size')  #
	parser.add_argument('--gnn_layer', default="[16,16]", type=str, help='gnn layers: number + dim')  #
	parser.add_argument('--time_slot', default=31104000, type=float, help='length of time slots')  #
	parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention')  #
	parser.add_argument('--gate_rate', default=0.8, type=float, help='gating rate')  #
	parser.add_argument('--point', default='ICDE_CIKM_WOCL', type=str, help='')
	parser.add_argument('--title', default='self_attention_behavior', type=str, help='title of model')

	#for train
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  #

	parser.add_argument('--opt_base_lr', default=1e-4, type=float, help='learning rate')
	parser.add_argument('--opt_max_lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--opt_weight_decay', default=1e-4, type=float, help='weight decay regularizer')

	parser.add_argument('--batch', default=4096, type=int, help='batch size')  #
	parser.add_argument('--reg', default=1.45e-2, type=float, help='weight decay regularizer')  #
	parser.add_argument('--epoch', default=1000, type=int, help='number of epochs')  #
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')  #
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	parser.add_argument('--mult', default=1, type=float, help='multiplier for the result')  #
	parser.add_argument('--drop_rate', default=0.1, type=float, help='drop_rate')  #
	parser.add_argument('--seed', type=int, default=19)  #
	parser.add_argument('--slope', type=float, default=0.1)  #
	parser.add_argument('--patience', type=int, default=300)
	parser.add_argument('--cl_long_rate', default=0.013, type=float, help='cl_rate')
	parser.add_argument('--cl_short_rate', default=0.00005, type=float, help='cl_rate')

	#for save and read
	parser.add_argument('--path', default='/home/ww/Code/MultiBehavior_BASELINE/MB-GCN/Datasets/', type=str, help='data path')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--dataset', default='IJCAI_15', type=str, help='name of dataset')  #
	parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	parser.add_argument('--isload', default=False , type=bool, help='whether load model')  #
	parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')

	parser.add_argument('--loadModelPath', default='/home/ww/Code/work1/master_behavior_attention/Model/IJCAI_15/ICDE_CIKM_WOCL_self_attention_behavior_IJCAI_15_2022_04_26__11_30_42_lr_0.001_reg_0.0145_batch_size_4096_time_slot_31104000.0_gnn_layer_[16,16].pth', type=str, help='loadModelPath')


	#use less
	# parser.add_argument('--memosize', default=2, type=int, help='memory size')
	parser.add_argument('--sampNum', default=10, type=int, help='batch size for sampling')  #
	# parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')  #
	# parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')  #
	parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')  #
	parser.add_argument('--iiweight', default=0.3, type=float, help='weight for ii')  #
	parser.add_argument('--graphSampleN', default=10000, type=int, help='use 25000 for training and 200000 for testing, empirically')  #
	parser.add_argument('--divSize', default=1000, type=int, help='div size for smallTestEpoch')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
	parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')  #
	parser.add_argument('--slot', default=0.5, type=float, help='length of time slots')  #
	parser.add_argument('--tau', default=0.035, type=float, help='')  #
	parser.add_argument('--positional_rate', default=0.0001, type=float, help='')  #
	#IJCAI_15------------------------------------------------------------------------------------------------------------------------------



	# #JD------------------------------------------------------------------------------------------------------------------------------
	# #for this model
	# parser.add_argument('--hidden_dim', default=16, type=int, help='embedding size')  #
	# parser.add_argument('--gnn_layer', default="[16,16,16]", type=str, help='gnn layers: number + dim')  #
	# parser.add_argument('--time_slot', default=7776000, type=float, help='length of time slots')  #
	# parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention')  #
	# parser.add_argument('--gate_rate', default=0.8, type=float, help='gating rate')  #
	# parser.add_argument('--point', default='ICDE_CIKM_WOCL', type=str, help='')
	# parser.add_argument('--title', default='self_attention_behavior', type=str, help='title of model')

	# #for train
	# parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  #

	# parser.add_argument('--opt_base_lr', default=1e-4, type=float, help='learning rate')
	# parser.add_argument('--opt_max_lr', default=1e-3, type=float, help='learning rate')
	# parser.add_argument('--opt_weight_decay', default=1e-4, type=float, help='weight decay regularizer')

	# parser.add_argument('--batch', default=4096, type=int, help='batch size')  #
	# parser.add_argument('--reg', default=1.45e-2, type=float, help='weight decay regularizer')  #
	# parser.add_argument('--epoch', default=1000, type=int, help='number of epochs')  #
	# parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')  #
	# parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	# parser.add_argument('--mult', default=1, type=float, help='multiplier for the result')  #
	# parser.add_argument('--drop_rate', default=0.1, type=float, help='drop_rate')  #
	# parser.add_argument('--seed', type=int, default=19)  #
	# parser.add_argument('--slope', type=float, default=0.1)  #
	# parser.add_argument('--patience', type=int, default=300)
	# parser.add_argument('--cl_long_rate', default=0.013, type=float, help='cl_rate')
	# parser.add_argument('--cl_short_rate', default=0.0005, type=float, help='cl_rate')

	# #for save and read
	# parser.add_argument('--path', default='/home/ww/Code/MultiBehavior_BASELINE/MB-GCN/Datasets/', type=str, help='data path')
	# parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	# parser.add_argument('--load_model', default=None, help='model name to load')
	# parser.add_argument('--dataset', default='JD', type=str, help='name of dataset')  #
	# parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	# parser.add_argument('--isload', default=False , type=bool, help='whether load model')  #
	# parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')

	# parser.add_argument('--loadModelPath', default='/home/ww/Code/work1/master_behavior_attention/Model/IJCAI_15/ICDE_CIKM_WOCL_self_attention_behavior_IJCAI_15_2022_04_26__11_30_42_lr_0.001_reg_0.0145_batch_size_4096_time_slot_31104000.0_gnn_layer_[16,16].pth', type=str, help='loadModelPath')


	# #use less
	# # parser.add_argument('--memosize', default=2, type=int, help='memory size')
	# parser.add_argument('--sampNum', default=10, type=int, help='batch size for sampling')  #
	# # parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')  #
	# # parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	# parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')  #
	# parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')  #
	# parser.add_argument('--iiweight', default=0.3, type=float, help='weight for ii')  #
	# parser.add_argument('--graphSampleN', default=10000, type=int, help='use 25000 for training and 200000 for testing, empirically')  #
	# parser.add_argument('--divSize', default=1000, type=int, help='div size for smallTestEpoch')
	# parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	# parser.add_argument('--subUsrSize', default=10, type=int, help='number of item for each sub-user')
	# parser.add_argument('--subUsrDcy', default=0.9, type=float, help='decay factor for sub-users over time')  #
	# parser.add_argument('--slot', default=0.5, type=float, help='length of time slots')  #
	# parser.add_argument('--tau', default=0.1, type=float, help='')  #
	# parser.add_argument('--positional_rate', default=0.0001, type=float, help='')  #
	# #JD------------------------------------------------------------------------------------------------------------------------------


	return parser.parse_args()
args = parse_args()

# args.user = 805506#147894
# args.item = 584050#99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
# args.user = 19800
# args.item = 22734

# swap user and item
# tem = args.user
# args.user = args.item
# args.item = tem

# args.decay_step = args.trn_num
# args.decay_step = args.item//args.batch
args.decay_step = args.trnNum//args.batch



# #----IJCAI_15-------------------------------------------------------------------------------------------------------------------------------------
# python ./main_ssl_dynamic.py  --dataset=IJCAI_15 --cl_long_rate=0.013 --cl_short_rate=0.00005 --tau=0.1 --time_slot=31104000 --gnn_layer=[16,16] --tau=0.035 --head_num=4
# [01:36:30] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: /home/ww/anaconda3/lib/python3.8/site-packages/dgl/tensoradapter/pytorch/libtensoradapter_pytorch_1.10.0.so: cannot open shared object file: No such file or directory
# Using backend: pytorch
# Namespace(batch=4096, cl_long_rate=0.015, cl_short_rate=5e-05, dataset='IJCAI_15', decay=0.96, decay_step=2, deep_layer=0, divSize=1000, drop_rate=0.1, epoch=1000, gate_rate=0.8, gnn_layer='[16,16]', graphSampleN=10000, head_num=4, hidden_dim=16, iiweight=0.3, isJustTest=False, isload=False, loadModelPath='/home/ww/Code/work1/master_behavior_attention/Model/IJCAI_15/topk_20_self_attention_behavior_IJCAI_15_2021_07_29__17_13_23_lr_0.001_reg_0.0145_batch_size_4096_time_slot_31104000_gnn_layer_[16,16].pth', load_model=None, lr=0.001, mult=1, opt_base_lr=0.0001, opt_max_lr=0.001, opt_weight_decay=0.0001, path='/home/ww/Code/MultiBehavior_BASELINE/MB-GCN/Datasets/', patience=300, point='topk_20', positional_rate=0.0001, reg=0.0145, sampNum=10, save_path='tem', seed=19, shoot=10, slope=0.1, slot=0.5, subUsrDcy=0.9, subUsrSize=10, target='buy', tau=0.1, time_slot=7776000.0, title='self_attention_behavior', trnNum=10000, tstEpoch=1)
# #----IJCAI_15-------------------------------------------------------------------------------------------------------------------------------------


# # #----Tmall-------------------------------------------------------------------------------------------------------------------------------------
# python ./main_ssl_dynamic.py  --dataset=Tmall --cl_long_rate=0.013 --cl_short_rate=0.0005 --tau=0.1  --time_slot=864000 --hidden_dim=16 --tau=0.035 --gnn_layer=[16,16,16,16]
# [09:08:13] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: /home/ww/anaconda3/lib/python3.8/site-packages/dgl/tensoradapter/pytorch/libtensoradapter_pytorch_1.10.0.so: cannot open shared object file: No such file or directory
# Using backend: pytorch
# Namespace(batch=4096, cl_long_rate=0.013, cl_short_rate=0.0005, dataset='Tmall', decay=0.96, decay_step=2, deep_layer=0, divSize=1000, drop_rate=0.1, epoch=1000, gate_rate=0.8, gnn_layer='[16,16,16,16]', graphSampleN=10000, head_num=4, hidden_dim=16, iiweight=0.3, isJustTest=False, isload=False, loadModelPath='/home/ww/Code/work1/master_behavior_attention/Model/IJCAI_15/topk_20_self_attention_behavior_IJCAI_15_2021_07_29__17_13_23_lr_0.001_reg_0.0145_batch_size_4096_time_slot_31104000_gnn_layer_[16,16].pth', load_model=None, lr=0.001, mult=1, opt_base_lr=0.0001, opt_max_lr=0.001, opt_weight_decay=0.0001, path='/home/ww/Code/MultiBehavior_BASELINE/MB-GCN/Datasets/', patience=300, point='topk_20', positional_rate=0.0001, reg=0.0145, sampNum=10, save_path='tem', seed=19, shoot=10, slope=0.1, slot=0.5, subUsrDcy=0.9, subUsrSize=10, target='buy', tau=0.035, time_slot=864000.0, title='self_attention_behavior', trnNum=10000, tstEpoch=1)
# # #----Tmall-------------------------------------------------------------------------------------------------------------------------------------

# python ./main_ssl_dynamic.py --cl_long_rate=0.013 --cl_short_rate=0.0005 --dataset=Tmall --decay=0.96 --deep_layer=0 --divSize=1000 --drop_rate=0.1 --epoch=1000 --gate_rate=0.8 --gnn_layer=[16,16,16,16] --graphSampleN=10000 --head_num=4 --hidden_dim=16 --iiweight=0.3 --lr=0.001 --mult=1 --opt_base_lr=0.0001 --opt_max_lr=0.001 --opt_weight_decay=0.0001 --patience=300 --positional_rate=0.0001 --reg=0.0145 --sampNum=10 --seed=19 --shoot=10 --slope=0.1 --slot=0.5 --subUsrDcy=0.9 --subUsrSize=10 --tau=0.035 --time_slot=864000 --trnNum=10000 --tstEpoch=1




# # #----JD-------------------------------------------------------------------------------------------------------------------------------------
# python ./main_ssl_dynamic.py  --dataset=JD --cl_long_rate=0.013 --cl_short_rate=0.0005 --tau=0.1 --time_slot=7776000 --gnn_layer=[16,16,16]
# [00:12:41] /opt/dgl/src/runtime/tensordispatch.cc:43: TensorDispatcher: dlopen failed: /home/ww/anaconda3/lib/python3.8/site-packages/dgl/tensoradapter/pytorch/libtensoradapter_pytorch_1.10.0.so: cannot open shared object file: No such file or directory
# Using backend: pytorch
# Namespace(batch=4096, cl_long_rate=0.013, cl_short_rate=0.0005, dataset='JD', decay=0.96, decay_step=2, deep_layer=0, divSize=1000, drop_rate=0.1, epoch=1000, gate_rate=0.8, gnn_layer='[16,16,16]', graphSampleN=10000, head_num=4, hidden_dim=16, iiweight=0.3, isJustTest=False, isload=False, loadModelPath='/home/ww/Code/work1/master_behavior_attention/Model/IJCAI_15/topk_20_self_attention_behavior_IJCAI_15_2021_07_29__17_13_23_lr_0.001_reg_0.0145_batch_size_4096_time_slot_31104000_gnn_layer_[16,16].pth', load_model=None, lr=0.001, mult=1, opt_base_lr=0.0001, opt_max_lr=0.001, opt_weight_decay=0.0001, path='/home/ww/Code/MultiBehavior_BASELINE/MB-GCN/Datasets/', patience=300, point='topk_20', positional_rate=0.0001, reg=0.0145, sampNum=10, save_path='tem', seed=19, shoot=10, slope=0.1, slot=0.5, subUsrDcy=0.9, subUsrSize=10, target='buy', tau=0.1, time_slot=7776000.0, title='self_attention_behavior', trnNum=10000, tstEpoch=1)
# # #----JD-------------------------------------------------------------------------------------------------------------------------------------
