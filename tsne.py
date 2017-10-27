import numpy as np
import pandas as pd
import os
from time import time
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
color = {
    0: (0, 0, 0),
    1: (1, 0.65, 0),
    2: (0, 1, 0),
    3: (0, 0.65, 1),
}

form = 'rf'

def loadData( filename ):
    if os.path.splitext( filename )[1] == '.csv':
        converter = {
            "target": lambda x: int( x )
        }
        use_cols = ["feat_" + str(i) for i in range(5)]
        use_cols.append("target")
        full = pd.read_csv( filename, sep=',', converters=converter, usecols=use_cols)
        full_mat = full.as_matrix()
        res = {
            "X": full_mat[:,:-1],
            "y": full_mat[:,-1]
        }
    return res

def random_slice_instance(X, y, n=None, idx=None):
    if n is None:
        return X,y
    elif n < 1:
        n = int( n * X.shape[0] )
    else:
        n = int(n)
    if idx is None:
        idx = np.array(range(X.shape[0]))
        np.random.shuffle(idx)
    X = X[idx[:n], :]
    y = y[idx[:n]]
    return X, y

def slice_instances_seperately( X, y, n_pos=None, n_neg=None,idx_p=None, idx_n=None ):
    neg_X = X[y == 0, :]
    neg_y = y[y == 0]
    pos_X = X[y == 1, :]
    pos_y = y[y == 1]
    pos_X, pos_y = random_slice_instance(pos_X, pos_y, n_pos, idx_p)
    neg_X, neg_y = random_slice_instance(neg_X, neg_y, n_neg, idx_n)
    X = np.concatenate( (pos_X, neg_X),axis=0)
    y = np.concatenate( (pos_y, neg_y))
    return X, y

def slice_instances_seperately1( X, y, n_pos=None, n_neg=None,idx_p=None, idx_n=None ):
    neg_X = X[y != 0, :]
    neg_y = y[y != 0]
    pos_X = X[y == 0, :]
    pos_y = y[y == 0]
    pos_X, pos_y = random_slice_instance(pos_X, pos_y, n_pos, idx_p)
    # neg_X, neg_y = random_slice_instance(neg_X, neg_y, n_neg, idx_n)
    X = np.concatenate( (pos_X, neg_X),axis=0)
    y = np.concatenate( (pos_y, neg_y))
    return X, y

def CV_stratified_slice(X, y,n_fold ):
    '''
    binary dataset cross validation stratified slice.
    :param X:
    :param y:
    :param n_fold:
    :return:
    '''
    t0 = time()
    n_neg = sum( y == 0)
    n_pos = sum( y == 1)
    idx_n = np.array( range(n_neg) )
    idx_p = np.array( range(n_pos) )
    np.random.shuffle(idx_n)
    np.random.shuffle(idx_p)
    n_pos_per_fold = int( n_pos/n_fold)
    n_neg_per_fold = int( n_neg/n_fold)
    res = []
    for i in range(n_fold):
        print( "slice : %s" %(i) )
        if i == ( n_fold -1 ):
            n_pos_per_fold = n_pos - n_pos_per_fold * ( n_fold - 1 )
            n_neg_per_fold = n_neg - n_neg_per_fold * ( n_fold - 1 )
            sliced_X, sliced_y = slice_instances_seperately(X,y,n_pos=n_pos_per_fold, n_neg=n_neg_per_fold, \
                                                           idx_p=idx_p[ -n_pos_per_fold:], idx_n=idx_n[-n_neg_per_fold:] )
        else:
            sliced_X, sliced_y = slice_instances_seperately(X, y, n_pos=n_pos_per_fold, n_neg=n_neg_per_fold, \
                                                            idx_p=idx_p[ i*n_pos_per_fold : (i+1)*n_pos_per_fold], idx_n=idx_n[i*n_neg_per_fold:(i+1)*n_neg_per_fold] )
        res.append({
            "X":sliced_X,
            "y":sliced_y
        })
    print( "cross validation dataset stratified slice done in %.2g sec" %( time()-t0 ) )
    return res

def dataset_concatenate( dataset ):
    X = None
    y = None
    for d in dataset:
        if X is None:
            X = d["X"]
            y = d["y"]
        else:
            X = np.concatenate( ( X, d["X"] ), axis=0 )
            y = np.concatenate( (y, d["y"] ) )
    return X, y

def plot_2D_tsne( X, y, name, ind ):
    n_components = 2

    #tsne transform
    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca')
    X_plat = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE:%.2g sec"%( t1 - t0 ) )

    #plot tsne result
    # fig = plt.figure(1)
    fig = plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)
    plt.scatter( X_plat[:,0], X_plat[:,1], c= [color[x + 1] for x in (y.astype(int))], edgecolors='face')
    np.savetxt('result-' + form + ind + '/Y' + str(name) + '.txt', X_plat)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    # idx_1 = ( y == 1 )
    # p1 = plt.scatter( X[idx_1,0], X[idx_1,1], marker = 'x', color='c', s=20, cmap=plt.cm.Spectral)
    # idx_2 = ( y == 0 )
    # p2 = plt.scatter( X[idx_2,0], X[idx_2,1], marker = 'o', color='r',cmap=plt.cm.Spectral)
    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.savefig('result-' + form + ind + '/result' + str(name) + '.png')
    plt.close()
    # plt.show()
    # return plt


def add_random_noise(X,y):
    for i in range( X.shape[0]):
        if abs(X[i,:].sum()) < 1e-6:
            y[i] = -1
            # X[i,:] += np.random.rand((X.shape[1]))/100000.0
    return X,y

def test_random_forest_data_from_changjian():
    err_set = [208]
    for i in range(0, 301):
        print(i)
        if i in err_set:
            continue
        try:
            data = loadData('pureTrainingData/model_' + str(i) + '_full.csv')
            n_samples = 2
            X = data["X"]
            y = data["y"]
            X, y = slice_instances_seperately(X, y, 1000)

            X, y = add_random_noise(X, y)
            plot_2D_tsne(X, y, i)
        except IOError:
            print('没找到文件')
        except IndexError:
            print('文件有问题')
        else:
            continue

def test_random_forest():
    err_set = [208]
    for i in range(0, 301):
        print(i)
        if i in err_set:
            continue
        try:
            X = np.load('tsne-data-' + form + '/' + form + str(i) + '.X.npy')
            X = X[:, 1:]
            y = np.load('tsne-data-' + form + '/' + form + str(i) + '.mark.npy')

            X, y = slice_instances_seperately1(X, y, 3000)

            X, y = add_random_noise(X, y)
            plot_2D_tsne(X, y, i, '')

            # data = loadData('pureTrainingData/model_' + str(i) + '_full.csv')
            # X = data["X"]
            # y = np.load('tsne-data-' + form + '/' + str(i) + '_rf.mark.npy')
            # X, y = slice_instances_seperately1(X, y, 3000)
            #
            # X, y = add_random_noise(X, y)
            # plot_2D_tsne(X, y, i, '-tran')
        except IOError:
            print('没找到文件')
        except IndexError:
            print('文件有问题')
        except AssertionError:
            print('文件有问题')
        except MemoryError:
            print('文件过大')
        else:
            print("成功")

def test_lightgbm():
    err_set = [208]
    for i in range(0, 301):
        print(i)
        if i in err_set:
            continue
        try:
            X_plat = np.load('tsne-data-lgb/' + str(i) + '_rf.X.npy')

            y = np.load('tsne-data-lgb/' + str(i) + '_rf.mark.npy')
            ax = plt.subplot(111)
            plt.scatter(X_plat[:, 0], X_plat[:, 1], c=[color[x + 1] for x in (y.astype(int))], edgecolors='face')
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            # idx_1 = ( y == 1 )
            # p1 = plt.scatter( X[idx_1,0], X[idx_1,1], marker = 'x', color='c', s=20, cmap=plt.cm.Spectral)
            # idx_2 = ( y == 0 )
            # p2 = plt.scatter( X[idx_2,0], X[idx_2,1], marker = 'o', color='r',cmap=plt.cm.Spectral)
            # ax.xaxis.set_major_formatter(NullFormatter())
            # ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
            plt.savefig('result-lgb/result' + str(i) + '.png')
            plt.close()
        except IOError:
            print('没找到文件')
        except IndexError:
            print('文件有问题')
        except AssertionError:
            print('文件有问题')
        else:
            print("成功")

if __name__ == '__main__':
    test_lightgbm()

    pass