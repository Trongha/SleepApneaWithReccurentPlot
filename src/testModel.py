import numpy as np
from src import config
from src import MyUtil as myUtil

from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report

print('done import')

recordNames = config.NAME_OF_RECORD
numRecord = len(recordNames)
for testRecordName in recordNames:
    recordNameForTest = [testRecordName]
    trainRqa, trainLabel, testRqa, testLabel = myUtil.loadAllRqa(recordNameForTest)
    print('done load RQA')
    #
    # print('trainRqa: ', trainRqa.shape)
    # # print(trainRqa[-10:])
    # print('trainLabel: ', trainLabel.shape)
    # print('testRqa: ', testRqa.shape)
    # # print(testRqa[-10:])
    # print('testLabel: ', testLabel.shape)
    #
    # unique, count = np.unique(trainLabel, return_counts=True)
    # print('trainLabel unique: ', dict(zip(unique, count)))
    unique, count = np.unique(testLabel, return_counts=True)
    print('testLabel unique: ', dict(zip(unique, count)))

    # ====================  FIT  ============================
    from sklearn import svm

    clf = svm.SVC(kernel='rbf')
    clf.fit(trainRqa, trainLabel)
    # print(clf)
    # print('done fit')

    # ====================  TEST  ============================
    pre = clf.predict(testRqa)
    # print('pre: ', pre)
    accuracy = accuracy_score(testLabel, pre)
    print("accuracy: {:.3f}, recall: {:.3f}, percision: {:.3f}".format(accuracy, recall_score(testLabel, pre), precision_score(testLabel, pre)))
    # print('recall: %.3f' % recall_score(testLabel, pre))
    # print('percision: %.3f' % precision_score(testLabel, pre))

    # print('confusion_matrix: \n', confusion_matrix(testLabel, pre))
    # print('classification_report: \n', classification_report(testLabel, pre))
    # print('done Test')


