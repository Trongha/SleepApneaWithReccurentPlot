from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import config as config

fileNormal = config.FOLDER_SAVE_RQA + 'normalData.npy'
fileApnea = config.FOLDER_SAVE_RQA + 'apneaData.npy'

rqaNormal = np.load(fileNormal)
rqaApnea = np.load(fileApnea)

numberOfNormal = rqaNormal.shape[0]
numberOfApnea = rqaApnea.shape[0]

numOfNormalTrain = int(numberOfNormal * 2 / 3)
numOfApneaTrain = int(numberOfApnea * 2 / 3)

rqaTrain = np.append(rqaNormal[0:numOfNormalTrain], rqaApnea[0:numOfApneaTrain], axis=0)
labelTrain = [config.NORMAL_LABEL] * numOfNormalTrain + [config.APNEA_LABEL] * numOfApneaTrain
labelTrain = np.array(labelTrain)
rqaTest = np.append(rqaNormal[numOfNormalTrain:], rqaApnea[numOfApneaTrain:], axis=0)
labelTest = [config.NORMAL_LABEL] * (numberOfNormal - numOfNormalTrain) \
            + [config.APNEA_LABEL] * (numberOfApnea - numOfApneaTrain)
labelTest = np.array(labelTest)

clf = MLPClassifier(activation='tanh', hidden_layer_sizes=(12, 50, 2), max_iter=500, alpha=0.0001,
                    solver='sgd', random_state=1, tol=0.000000001)

print(clf)
clf.fit(rqaTrain, labelTrain)

pre = clf.predict(rqaTest)
cc = accuracy_score(labelTest, pre)
print(cc)

print('stop')

print('stop')
