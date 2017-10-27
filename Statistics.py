import numpy as np
import pandas
import csv

notUsedNum = [18,187]
wrongFormNum = [21,48,78,79,111,142,169,170,178,189,205,220,257,283,289,295]

def pos_neg_num(filename):
    pos = 0
    neg = 0
    with open(filename) as f:
        ls = csv.reader(f)
        header = next(f)
        for line in ls:
            if float(line[6]) < 0.5:
                neg += 1
            else:
                pos += 1
    return pos, neg

if __name__ == '__main__':
    Pos = []
    Neg = []
    Acc = []
    # for i in range(301):
    for i in [9]:
        if i in notUsedNum or i in wrongFormNum:
            continue
        pos, neg = pos_neg_num( '../../Ximei/pureTrainingData/model_' + str(i)+ '_full.csv')
        acc = float(pos)/(neg+pos)
        Pos.append(pos)
        Neg.append(neg)
        Acc.append(acc)
        print( pos, neg, acc)
    Acc = np.array( Acc )
    Neg = np.array( Neg )
    Pos = np.array( Pos )
    print( "Acc:",Acc.mean(), np.array(Acc) )
    print( "Neg:",Neg.mean() )
    print( "Pos:",Pos.mean() )
