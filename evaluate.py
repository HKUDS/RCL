import numpy as np
import torch

def hit(gt_item, pred_items):
    """
    正常的算法: HR----(分子: topK中商品的个数)/(分母: 测试集中商品的个数)
    TODO: 具体怎么算, 要传入的参数?
    :param gt_item:
    :param pred_items:
    :return:
    """
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    """
    具体的计算: 其实这里ndcg的计算和师兄在公式上的稍有不同: 因为这里直接取"位置的倒数"所以, 相当于公式中的分子==1
    :return:
    """
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+1))  #师兄在这个算的是2, 但是我认为是1
    return 0

def metrics(model, test_loader, top_k):
    """
    作用: 这个函数只是在这里定义了一下, 在整个project中也没有去调用
    :param model:
    :param test_loader:
    :param top_k:
    :return:
    """
    HR, NDCG = [], []

    for user, item_i, item_j in test_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()

        prediction_i, prediction_j = model(user, item_i, item_j)  #获得所有, 正样本, 负样本的预测值
        _, indices = torch.topK(prediction_i, top_k) #因为原本的数组可能是乱序的, 所以找出最大或者最小值. 意义: 预测值最大的就是最有可能的
        recommends = torch.take(item_i, indices).cpu().numpy().tolist()

        gt_item = item_i[0].item()  # TODO: 为什么这里取真值中的第一个, 虽然好像没有调用, 但是度量指标到时候要好好看一下
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

    

    