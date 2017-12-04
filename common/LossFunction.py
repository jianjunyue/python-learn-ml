import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

#自定义损失函数

class LossFunction(object):

    # 自定义损失函数评分
    # 实际为0，预测为1，loss 10分
    # 实际为1，预测为0，loss 2分
    # @pre_y：预测的结果
    # @test_y:实际的结果
    def loss_score(pre_y,test_y):
        loss_error = 0;
        z = zip(pre_y, test_y)
        for py, ty in z:
            if (py != ty):
                if (ty == 0):
                    loss_error += 10
                else:
                    loss_error += 2
        return loss_error


    #对于二分类问题，返回真正的否定，假阳性，假阴性和真阳性的计数
    def binaryTNFPFNTP(self,y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return  tn, fp, fn, tp

