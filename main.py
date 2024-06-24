import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
import xgboost as xgb
import random
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(os.getcwd())



def SL_data():
    sl_p = np.loadtxt('./data/SL.txt', dtype=np.int64,delimiter=',')
    sl_p = np.insert(sl_p, 2, np.ones(len(sl_p)), axis=1)
    sl_n = np.loadtxt('./data/yard.txt',delimiter='\t', dtype=np.int64)
    sl_n = np.insert(sl_n, 2, np.zeros(len(sl_n)), axis=1)
    sl = np.concatenate((sl_p,sl_n),axis=0)

    return sl


def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc


def pr_auc(y, pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)

    return pr_auc


def scores(y, pred):
    scores = pred
    scores[scores >= 0.5] = 1
    scores[scores < 0.5] = 0
    f1 = metrics.f1_score(y_true=y, y_pred=pred)
    acc = metrics.accuracy_score(y_true=y, y_pred=pred)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for j in range(len(y)):
        if y[j] == 1:
            if y[j] == pred[j]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y[j] == pred[j]:
                tn = tn + 1
            else:
                fp = fp + 1
    if tp == 0 and fp == 0:
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        precision = 0
        MCC = 0
    else:
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        precision = float(tp) / (tp + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))

    Pre = np.float64(precision)
    Sen = np.float64(sensitivity)
    Spe = np.float64(specificity)
    MCC = np.float64(MCC)
    f1 = np.float64(f1)
    acc = np.float64(acc)

    return f1, acc, Pre, Sen, Spe, MCC


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def get_features(data,ppi_emb,use_pro):
    genea_features = pd.merge(data,ppi_emb,how='left',left_on='head',right_on='name').iloc[:, 5:261].values
    geneb_features = pd.merge(data, ppi_emb, how='left', left_on='tail', right_on='name').iloc[:, 5:261].values
    if use_pro:
        feature = np.concatenate([genea_features,geneb_features],axis=1)
        feature = mms.fit_transform(feature)
    else:
        feature = genea_features
    return feature


def get_embedding(train_d,test_d,get_scaled,n_components):
    saver = tf.train.import_meta_graph('./model/KGCN_1.ckpt.meta')
    with tf.Session() as sess:
        saver.restore(sess, "./model/KGCN_1.ckpt")
        var_list = tf.global_variables()  # 获取图上的所有变量
        entity_emb = [v for v in var_list if "entity_emb_matrix/Adam_1:0" in v.name]
        for v in entity_emb:
            # print(v, sess.run(v)[1])
            entity_emb_matrix=sess.run(v)

        train_gene1 = tf.nn.embedding_lookup(entity_emb_matrix, train_d['head'])
        train_gene2 = tf.nn.embedding_lookup(entity_emb_matrix, train_d['tail'])
        test_gene1 = tf.nn.embedding_lookup(entity_emb_matrix, test_d['head'])
        test_gene2 = tf.nn.embedding_lookup(entity_emb_matrix, test_d['tail'])

        train_gene1=train_gene1.eval()
        train_gene2=train_gene2.eval()
        test_gene1=test_gene1.eval()
        test_gene2=test_gene2.eval()

    train_feat=np.concatenate([train_gene1,train_gene2],axis=1)
    test_feat=np.concatenate([test_gene1,test_gene2],axis=1)
    train_feat=mms.fit_transform(train_feat)
    test_feat=mms.fit_transform(test_feat)
    if get_scaled:
        pca = PCA(n_components=n_components)
        train_feat = pca.fit_transform(train_feat)  
        test_feat = pca.transform(test_feat)
    else:
        train_feat = train_feat
        test_feat = test_feat

    return train_feat,test_feat


def train(embedding_dim, n_components, use_pro, patience, train_data, test_data,get_scaled):
    train_dense_features, test_dense_features= get_embedding(train_data, test_data,get_scaled,n_components)
    train_label = train_data['label'].values
    test_label = test_data['label'].values
    train_des = get_features(train_data, ppi_emb, use_pro)
    test_des = get_features(test_data, ppi_emb, use_pro)
    train_all_feats = np.concatenate([train_dense_features,train_des],axis=1)
    test_all_feats = np.concatenate([test_dense_features,test_des],axis=1)
    train_all_feats_scaled = mms.fit_transform(train_all_feats)
    test_all_feats_scaled = mms.transform(test_all_feats)

    # 调用xgb中的DMatrix()函数，把数据格式转换为xgb需要的模式
    dtrain = xgb.DMatrix(train_all_feats_scaled, label=train_label)
    dtest = xgb.DMatrix(test_all_feats_scaled)

    # 参数准备
    params = {'booster': 'gbtree',  # 弱学习器的类型，默认就是gbtree，及cart决策树
              'objective': 'binary:logistic',  # 目标函数，二分类：逻辑回归，输出的是概率
              'eval_metric': 'error',
              'max_depth': 12,  # 最大深度
              'lambda': 0.5,
              'subsample':0.75,
              'colsample_bytree':0.9,
              'eta': 0.1,  # 步长
              'seed': 0,
              'verbosity': 1,
              'device': 'cuda'}

    watchlist = [(dtrain,"train")]
    # 开始训练模型
    # params是传入模型的各个参数，以字典的形式传入
    model = xgb.train(params,
                      dtrain,
                      num_boost_round=100,  # 迭代的次数，及弱学习器的个数
                      evals=watchlist)

    # 对测试集合预测
    pred = model.predict(dtest)
    # 模型预测
    roc_xgb = roc_auc(test_label, pred)
    pr_xgb = pr_auc(test_label, pred)
    f1_xgb, acc_xgb, Pre_xgb, Sen_xgb, Spe_xgb, MCC_xgb = scores(test_label, pred)
    print('%.4f  %.4f   %.4f %.4f  %.4f  %.4f  %.4f   %.4f' %(roc_xgb, pr_xgb, f1_xgb, acc_xgb, Pre_xgb, Sen_xgb, Spe_xgb, MCC_xgb))

    return roc_xgb, pr_xgb,f1_xgb, acc_xgb, Pre_xgb, Sen_xgb, Spe_xgb, MCC_xgb, train_label, test_label,pred


def cross_validation(K_fold, examples):

    ROC_xgb = []
    PR_xgb = []
    F1_xgb = []
    ACC_xgb = []
    PRE_xgb = []
    SEN_xgb = []
    SPE_xgb = []
    MCC_xgb = []
    loss_curve = pd.DataFrame(
        columns=['step', 'ROC_nfm', 'PR_nfm', 'F1_nfm', 'Acc_nfm', 'Pre_nfm', 'Sen_nfm', ' Spe_nfm', 'MCC_nfm'])
    step = 1
    subsets = dict()
    n_subsets = int(len(examples) / K_fold)
    remain = set(range(0, len(examples) - 1))
    for i in reversed(range(0, K_fold - 1)):
        subsets[i] = random.sample(remain, n_subsets)
        remain = remain.difference(subsets[i])
    subsets[K_fold - 1] = remain

    for i in reversed(range(0, K_fold)):
        test_data = examples[list(subsets[i])]
        train_data = []
        for j in range(0, K_fold):
            if i != j:
                train_data.extend(examples[list(subsets[j])])
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        train_data.columns = ['head', 'tail', 'label']
        test_data.columns = ['head', 'tail', 'label']
        roc_xgb, pr_xgb, f1_xgb, acc_xgb, Pre_xgb, Sen_xgb, Spe_xgb, Mcc_xgb, train_label, test_label, pred_y = train(100, 200, True, 5, train_data, test_data, False)
        loss_curve.loc[step] = [step, roc_xgb, pr_xgb, f1_xgb, acc_xgb, Pre_xgb, Sen_xgb, Spe_xgb, Mcc_xgb]

        ROC_xgb.append(roc_xgb)
        PR_xgb.append(pr_xgb)
        F1_xgb.append(f1_xgb)
        ACC_xgb.append(acc_xgb)
        PRE_xgb.append(Pre_xgb)
        SEN_xgb.append(Sen_xgb)
        SPE_xgb.append(Spe_xgb)
        MCC_xgb.append(Mcc_xgb)
        step += 1

    loss_curve.loc[step] = [77, np.mean(ROC_xgb), np.mean(PR_xgb),np.mean(F1_xgb), np.mean(ACC_xgb), np.mean(PRE_xgb), np.mean(SEN_xgb), np.mean(SPE_xgb), np.mean(MCC_xgb)]
    print("nfm模型的评估参数为", 'ROC_nfm_mean %.4f' % np.mean(ROC_xgb), 'PR_nfm_mean %.4f' % np.mean(PR_xgb), 'F1_nfm_mean %.4f' % np.mean(F1_xgb), 'ACC_nfm_mean %.4f' % np.mean(ACC_xgb), 'PRE_nfm_mean %.4f' % np.mean(PRE_xgb), 'SEN_nfm_mean %.4f' % np.mean(SEN_xgb), 'SPE_nfm_mean %.4f' % np.mean(SPE_xgb), 'MCC_nfm_mean %.4f' % np.mean(MCC_xgb))


if __name__ =="__main__":
    # ppi future
    ppi_emb = pd.read_csv('./data/ppi_DeepWalk.txt')
    # sl
    sl = SL_data()
    kf = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    np.random.shuffle(sl)
    mms = MinMaxScaler(feature_range=(0, 1))
    cross_validation(5, sl)
    print('finish')
