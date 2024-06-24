from model import KGCN
import argparse
import warnings
import tensorflow as tf
from data_loader import *
from sklearn.model_selection import train_test_split,ShuffleSplit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


np.random.seed(555)
parser = argparse.ArgumentParser()
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use sum concat neighbor')
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=128, help='dimension of user and entity embeddings')
parser.add_argument('--n_hop', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--l2_weight', type=float, default=0.0039, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--trainv1_testv2', type=bool, default=False, help='train_on_v1_test_on_v2')
parser.add_argument('--earlystop_flag', type=bool, default=True, help='whether early stopping')
parser.add_argument('--rd', type=bool, default=True, help='whether use du test')
parser.print_help()


param_name = 'final'  # set as the parameter name while adjust parameters
args = parser.parse_args()
data = load_data(args)  # n_nodea, n_nodeb, n_entity, n_relation, adj_entity（4）, adj_relation（5）, isolated_point（6）


def cross_validation(K_fold,examples,neighbor_sample_size,data):
    np.random.shuffle(examples)
    a=0

    test_auc_kf_list = []
    test_aupr_kf_list = []
    test_f1_kf_list = []
    test_acc_kf_list = []
    test_pre_kf_list = []
    test_sen_kf_list = []
    test_spe_kf_list = []
    test_mcc_kf_list = []
    subsets=dict()
    n_subsets=int(len(examples)/K_fold)
    remain=set(range(0,len(examples)-1))
    for i in reversed(range(0,K_fold-1)):
        subsets[i]=random.sample(remain,n_subsets)
        remain=remain.difference(subsets[i])
    subsets[K_fold-1]=remain
    for i in reversed(range(0,K_fold)):
        test_d=examples[list(subsets[i])]
        eval_data,test_data=train_test_split(test_d,test_size=0.5)
        train_d=[]
        for j in range(0,K_fold):
            if i!=j:
                train_d.extend(examples[list(subsets[j])])
        train_data=np.array(train_d)
        a+=1
        n_nodea, n_nodeb, n_entity, n_relation = data[0], data[1], data[2], data[3]
        adj_entity, adj_relation = data[4], data[5]
        tf.reset_default_graph()
        model = KGCN(args, n_entity, n_relation, adj_entity, adj_relation)
        cross_validation = 1
        loss_kf_list = []
        loss_curve = pd.DataFrame(columns=['epoch', 'loss', 'test_auc', 'test_f1', 'test_acc', 'test_aupr', 'test_Pre', 'test_Sen', 'test_Spe', 'test_MCC'])
        kk = 1

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            best_loss_flag = 1000000
            early_stopping_flag = 2
            best_eval_auc_flag = 0
            for step in range(args.n_epochs):
                # training
                loss_list = []
                start = 0
                # skip the last incomplete minibatch if its size < batch size
                while start + args.batch_size <= train_data.shape[0]:
                    _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                    start += args.batch_size
                    loss_list.append(loss)
                loss_mean = np.mean(loss_list)

                train_auc,train_aupr,train_f1, train_acc, train_Pre, train_Sen, train_Spe, train_MCC = ctr_eval(sess, args, model, train_data,args.batch_size)
                eval_auc, eval_aupr,eval_f1, eval_acc,  eval_Pre, eval_Sen, eval_Spe, eval_MCC = ctr_eval(sess, args, model, eval_data, args.batch_size)
                test_auc, test_aupr,test_f1, test_acc,  test_Pre, test_Sen, test_Spe, test_MCC = ctr_eval(sess, args, model, test_data,args.batch_size)
                print('%.2sstep %.4f  %.4f   %.4f %.4f  %.4f  %.4f  %.4f   %.4f' %(step,test_auc, test_aupr,test_f1, test_acc,  test_Pre, test_Sen, test_Spe, test_MCC))
                loss_curve.loc[step] = [step, loss_mean, test_auc, test_aupr,test_f1, test_acc,  test_Pre, test_Sen, test_Spe, test_MCC]

                if (eval_auc > best_eval_auc_flag):

                    best_eval_auc_flag = eval_auc
                    best_test_auc = test_auc
                    best_test_aupr = test_aupr
                    best_test_f1 = test_f1
                    best_test_acc = test_acc
                    best_test_Pre = test_Pre
                    best_test_Sen = test_Sen
                    best_test_Spe = test_Spe
                    best_test_MCC = test_MCC

                    saver.save(sess,'./model/KGCN' + '_' + str(a) + '.ckpt')

                # early_stopping
                if args.earlystop_flag:
                    if loss_mean < best_loss_flag:
                        stopping_step = 0
                        best_loss_flag = loss_mean
                    else:
                        stopping_step += 1
                        if stopping_step >= early_stopping_flag:
                            print('Early stopping is trigger at step: %.4f  loss: %.4f  test_auc: %.4f  test_f1: %.4f   test_aupr: %.4f' % (step, loss_mean, test_auc, test_f1, test_aupr))
                            break

        test_auc_kf_list.append(best_test_auc)
        test_aupr_kf_list.append(best_test_aupr)
        test_f1_kf_list.append(best_test_f1)
        test_acc_kf_list.append(best_test_acc)
        test_pre_kf_list.append(best_test_Pre)
        test_sen_kf_list.append(best_test_Sen)
        test_spe_kf_list.append(best_test_Spe)
        test_mcc_kf_list.append(best_test_MCC)

        loss_kf_list.append(loss)
        cross_validation = cross_validation + 1

    loss_kf_mean = np.mean(loss_kf_list)
    print(loss_kf_mean)
    print('%.4f  %.4f   %.4f %.4f  %.4f  %.4f  %.4f   %.4f' %(np.mean(test_auc_kf_list), np.mean(test_aupr_kf_list), np.mean(test_f1_kf_list), np.mean(test_acc_kf_list), np.mean(test_pre_kf_list), np.mean(test_sen_kf_list), np.mean(test_spe_kf_list), np.mean(test_mcc_kf_list)))


def get_feed_dict(model, data, start, end):
    feed_dict = {model.nodea_indices: data[start:end, 0],
                 model.nodeb_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, args, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []

    acc_list = []
    aupr_list = []
    Pre_list = []
    Sen_list = []
    Spe_list = []
    MCC_list = []

    while start + batch_size <= data.shape[0]:
        auc, aupr,f1, acc,  Pre, Sen, Spe, MCC = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))

        auc_list.append(auc)
        f1_list.append(f1)
        acc_list.append(acc)
        aupr_list.append(aupr)
        Pre_list.append(Pre)
        Sen_list.append(Sen)
        Spe_list.append(Spe)
        MCC_list.append(MCC)

        start += batch_size
    return float(np.mean(auc_list)),float(np.mean(aupr_list)), float(np.mean(f1_list)), float(np.mean(acc_list)),  float(np.mean(Pre_list)), float(np.mean(Sen_list)), float(np.mean(Spe_list)), float(
        np.mean(MCC_list))


if __name__ =="__main__":
    cross_validation(5,data[6],args.neighbor_sample_size,data)