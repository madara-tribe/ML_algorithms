from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb # https://qiita.com/taki_tflare/items/dfa47e3f353baf96670b
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import datasets, metrics, cross_validation
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as c_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
from matplotlib.pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 12, 4


# In[44]:


f1=pd.read_csv('train_db.csv')
f1=f1.fillna(0)
f1=f1.drop('Unnamed: 0', axis=1)
f1=f1.drop('kaiin_id', axis=1)
f1=f1.drop('tori_id', axis=1)
f1=f1.drop('brand_name', axis=1)
f1=f1.drop('cate_name', axis=1)
f1=f1.drop('model_id', axis=1)
f1=f1.drop('cate_id', axis=1)
f1=f1.drop('brand_id', axis=1)
print(f1.shape)
f1.head()


# In[46]:


# ラベル
t=np.array(f1['label'])
t=[int(i-1) for i in t]
t=np.reshape(t, (len(t),))
print(t.shape)

# 特徴量
data=f1.drop('label', axis=1)
print(data.shape)

# 標準化
f1_columns=list(data.columns)
sc = StandardScaler()
dff = sc.fit_transform(data)
dff=pd.DataFrame(dff)
dff.columns=f1_columns
dff.head()


# train/test データに分ける
train_x, test_x, train_y, test_y = train_test_split(data,
                                t, test_size=0.1, random_state=0)


"""
ハイパーパラメータ探索
"""

# デフォルト値のパラメーターで特徴量の重要度を可視化

def modelfit(alg, train_x, train_y, test_x, test_y,
             useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_x, label=train_y)
        xgtest = xgb.DMatrix(test_x)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train_x, train_y, eval_metric='auc')
    
        
    #Predict training set:
    train_predictions = alg.predict(train_x)
    dtrain_predprob = alg.predict_proba(train_x)[:,1]
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, dtrain_predprob.round()))

    #  Predict on testing data:
    dtest_predprob = alg.predict_proba(test_x)[:,1]
                
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


# In[ ]:


xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=6,
        min_child_weight=1,
        objective= "multi:softmax",
        num_class=6,
        nthread=4,
        scale_pos_weight=1,
        seed=27)
modelfit(xgb1, train_x, train_y, test_x, test_y)

# データを行列に変換
train_x=np.reshape(np.array(train_x), (len(train_x),19))
test_x=np.reshape(np.array(test_x), (len(test_x),19))
train_y=np.reshape(np.array(train_y), (len(np.array(train_y)),))
test_y=np.reshape(np.array(test_y), (len(np.array(test_y)),))


# In[24]:


def objective(params):

    skf = cross_validation.StratifiedKFold(
        train_y, # Samples to split in K folds
        n_folds=5, # Number of folds. Must be at least 2.
        shuffle=True, # Whether to shuffle each stratification of the data before splitting into batches.
        random_state=30 # pseudo-random number generator state used for shuffling
    )

    boost_rounds = []
    score = []

    for train, test in skf:
        _train_x, _test_x, _train_y, _test_y =             train_x[train], train_x[test], train_y[train], train_y[test]

        train_xd = xgb.DMatrix(_train_x, label=_train_y)
        test_xd = xgb.DMatrix(_test_x, label=_test_y)
        watchlist = [(train_xd, 'train'),(test_xd, 'eval')]

        model = xgb.train(
            params,
            train_xd,
            num_boost_round=100,
            evals=watchlist,
            early_stopping_rounds=30
        )

        boost_rounds.append(model.best_iteration)
        score.append(model.best_score)

    print('average of best iteration:', np.average(boost_rounds))
    return {'loss': np.average(score), 'status': STATUS_OK}

def optimize(trials):
    space = {'booster':'dart',
         'learning_rate':0.1,
         'n_estimators':1000,
         'sample_type':'uniform',
         'normalize_type': 'tree',
         'objective': "multi:softmax",
         'min_child_weight':1,
         'max_depth':9,
         'eval_metric': "mlogloss",
         'num_class' : 6,
         'nthread':4,
         'scale_pos_weight':1,
         'seed':27,}
    
    # minimize the objective over the space
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=2
    )

    return best_params


# In[ ]:


trials = Trials()
best_params = optimize(trials)
print(best_params)

"""
ax_depth=5
[03:47:22] /workspace/src/gbm/gbtree.cc:494: drop 0 trees, weight = 1
[99]	train-mlogloss:0.002439	eval-mlogloss:0.002736
average of best iteration: 99.0
"""


# 損失関数(loss)の計算
print(objective(best_params))

"""
[99]	train-rmse:0.037219	eval-rmse:0.041508
average of best iteration: 99.0
{'loss': 0.041645, 'status': 'ok'}
"""


# trainとbestパラメーターを求める
train_xd = xgb.DMatrix(train_x, label=train_y)
bst = xgb.train(best_params, train_xd, num_boost_round=100)

# MSE
pred_y = bst.predict(xgb.DMatrix(test_x))
mse = metrics.mean_squared_error(test_y, pred_y)
print(mse)


"""
ハイパーパラメータの可視化.etc
"""
# 訓練後、特徴量の重要度の可視化
imp=bst.get_fscore()
xgb.plot_importance(bst)


# 木の可視化と各枝のlabelの内訳
xgb.to_graphviz(bst)


# 木をpng fileで保存
graph=xgb.to_graphviz(bst)
graph.format='png'
graph.render('tree1')


# 念のため、不均衡データに適してるMCCで評価（正解率100%）

from sklearn.metrics import matthews_corrcoef
thresholds = np.linspace(0.01, 0.99, 50)
mcc = np.array([matthews_corrcoef(test_y, pred_y>thr) for thr in thresholds])
plt.plot(thresholds, mcc)
best_threshold = thresholds[mcc.argmax()]
print(mcc.max())

# モデルの保存
import joblib
#save model
joblib.dump(bst, 'hyp.model') 

# モデルのロード
xgb = joblib.load('hyp.model')
