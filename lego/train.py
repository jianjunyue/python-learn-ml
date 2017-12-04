import xgboost as xgb
import numpy as np
import sys
import datetime

data = xgb.DMatrix("legao.out")
slice_index = int(data.num_row() * 9 / 10)
data_train = data.slice(range(0, slice_index, 1))
data_test = data.slice(range(slice_index + 1, data.num_row(), 1))

# print(type(data_train))


param = {'max_depth': 6,
         'eta': 0.2,
         'silent': 1,
         'objective': 'binary:logistic',
         'booster': 'gbtree',
         'nthread': 45}
label = data.get_label()
param['scale_pos_weight'] = float(np.sum(label == 0)) / np.sum(label == 1)
param['eval_metric'] = ['auc']
num_round = 50
bst = xgb.train(param, data_train, num_round, [(data_train, 'train'), (data_test, 'eval')])
bst.save_model('xg.model.'+ str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
print(sorted(bst.get_score(importance_type="gain").items(), key=lambda e: e[1], reverse=True))