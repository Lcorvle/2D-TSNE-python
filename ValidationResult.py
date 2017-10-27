import numpy as np
import os

def str2list( s ):
    res = s.strip().split(',')
    res = [float(i) for i in res]
    return res

def ten_cv_result( filename ):
    res = open(filename,'r')
    accuracy = str2list(res.readline())
    precision = str2list(res.readline())
    recall = str2list(res.readline())
    return accuracy, precision, recall


if __name__ == '__main__':
    # print( ten_cv_result( '../../Ximei/validationResult/0_validation.csv') )
    res = []
    acc = []
    pre = []
    rec = []
    for i in range(301):
        filename = '../../Ximei/validationResult/' + str(i) + '_validation.csv'
        try:
            accuracy, precision, recall = ten_cv_result(filename)
            acc.append( np.array(accuracy).mean()  )
            pre.append(np.array(precision).mean())
            rec.append(np.array(recall).mean())
            sum = np.array(accuracy).sum() + np.array(precision).sum() + np.array(recall).sum()
            if( sum < 30.0 ):
                res.append(i)
                print(i, np.array(accuracy).mean(), np.array(precision).mean(), np.array(recall).mean() )
        except:
            acc.append( 0 )
            pre.append( 0 )
            rec.append( 0 )
            print( "%s not found!"%(i))
    print(len(res))
    acc = np.array(acc)
    pre = np.array(pre)
    rec = np.array(rec)
    print( acc.mean())
    print( pre.mean())
    print( rec.mean())