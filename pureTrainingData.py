import csv
import os
import numpy as np
from sklearn.externals import joblib

solderSpec = {}
cluster = {}
col_types = [float, float, float, float, float]
with open( '../solder_spec.csv' ) as f:
    ls = csv.reader(f)
    # headers = next(f)
    for line in ls:
        sld = line[1] + '_' +line[2]
        raw = tuple(convert(value) for convert, value in zip(col_types, line[4:9]))
        raw = np.array(raw)
        tmp = raw.copy()
        raw[:3] = tmp[2:]
        raw[3:] = tmp[:2]
        solderSpec[sld] = raw

        cluster[sld] = line[-6]
    print("making dictionary ... complete!")

id = 0
map = {
    "FAIL":1,
    "PASS":2
}


# for i in range(39,301):
for i in [81]:
    train_data = []
    id = 0
    if i == 187 or i == 18:
        continue
    filename = '../../Ximei/clusterData/' + str(i) + '_group_all.csv'
    print(filename)
    # try:
    with open(filename) as f:
        ls = csv.reader(f)
        headers = next(f)
        for line in ls:
            fdate = line[0]
            y = map[line[3]]
            try:
                sld = line[1] + '_' +str(int(float(line[2])))
            except:
                sld = line[1] + '_' + line[2]
            row = tuple(convert(value) for convert, value in zip(col_types, line[4:9]))
            row = np.array(row)
            if sld in solderSpec:
                id = id + 1
                sp = solderSpec[sld]
                sp = np.array(sp)
                sp[0] *= 1000000000
                sp[1] *= 1000000
                sp[2] *= 1000
                X = row / sp
                train_data.append(
                    [id, X[0], X[1], X[2], X[3], X[4], str(y)]
                )
    # except:
    #     print(filename, "some wrong")
    #     continue
    resultPath = '../../Ximei/pureTrainingData/model_'+ str(i) +'_all.csv'
    with open(resultPath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','feat_0','feat_1','feat_2','feat_3','feat_4','target'])
        writer.writerows(train_data)
    csvfile.close()

