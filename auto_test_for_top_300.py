import csv
import os
import numpy as np
from sklearn.externals import joblib

modelDir = "model/"
clusterPath = "solder_to_classifier"
logDir = "groupdata/"


# load the model
clusterModels = {}
files= os.listdir(modelDir)
for file in files:
    if not os.path.isdir(file):
        modelPath = modelDir + file
        print( modelPath )
        sld = file.split('_model.txt')[0]
        model = joblib.load(modelPath)
        clusterModels[sld] = model

# ------------------------- load spec --------------------------
solderSpec = {}
cluster = {}
col_types = [float, float, float, float, float]
with open( 'solder_spec.csv' ) as f:
    ls = csv.reader(f)
    headers = next(f)
    for line in ls:
        sld = line[0] + '_' +line[1]
        solderSpec[sld] = tuple(convert(value) for convert, value in zip(col_types, line[3:8]))
        cluster[sld] = line[-6]
    print("making dictionary ... complete!")

BoardResult = {}
SPIFAILSolders = []
files= os.listdir(logDir)
for file in files:
    if not os.path.isdir(file):
        logPath = logDir + file
        print("")
        print(logPath)
        #--------------get basic info of Board--------------
        with open(logPath) as f:
            ls = csv.reader(f)
            headers = next(f)
            for line in ls:
                fdate = line[0]
                y = line[3]
                sld = line[1] + '_' + line[2]
                if(y == 'PASS'):
                    if fdate in BoardResult:
                        continue
                    else:
                        BoardResult[fdate] = 'PASS'
                    continue
                row = tuple(convert(value) for convert, value in zip(col_types, line[4:9]))
                row = np.array(row)
                if sld in solderSpec:
                    sp = solderSpec[sld]
                    sp = np.array(sp)
                    sp[0] *= 1000000000
                    sp[1] *= 1000000
                    sp[2] *= 1000
                    X = row / sp
                    if sld in cluster:
                        index = cluster[sld]
                        if index in clusterModels:
                            model = clusterModels[index]
                            tempData = np.vstack((X, X))
                            prediction_label = model.predict(tempData)[0]
                            if (prediction_label == 'FAIL'):
                                BoardResult[fdate] = 'FAIL'
                                SPIFAILSolders.append(
                                    [line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8],
                                     'FAIL','FAIL'])
                                print("FAIL")
                            else:
                                SPIFAILSolders.append(
                                    [line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8],
                                     'FAIL','PASS'])
                                print("PASS")
                                if fdate in BoardResult:
                                    continue
                                else:
                                    BoardResult[fdate] = 'PASS'
resultPath = 'autoTestBoardResult.csv'
with open(resultPath, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fdate','BoardStatus'])
    for key in BoardResult:
        rsl = [key,BoardResult[key]]
        writer.writerow(rsl)
csvfile.close()
resultPath = 'SPIFAILData.csv'
with open(resultPath, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['fdate','compname','WinName','status','v','a','h','px','py','SPIStatus','AutoTestStatus'])
    writer.writerows(SPIFAILSolders)
csvfile.close()








