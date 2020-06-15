import os
from typing import Union, Iterable
from tqdm import tqdm

import numpy as np
from src import RecurrentPlot as rp
from numpy.core._multiarray_umath import ndarray

from src import config as config


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def getLabelAndInfo(recordName='a01', type='train'):
    labelFile = config.getFileSaveLabel(recordName, type)
    infoFile = config.getFileSaveInfo(recordName, type)
    label = np.load(labelFile, allow_pickle=True)
    info = np.load(infoFile, allow_pickle=True)
    return label, info


def readRpBinary(recordName='a01', type='train'):
    # todo: use getFileName
    dataFile = config.getFileSaveRp(recordName, type)
    listDotOfRp = np.load(dataFile, allow_pickle=True)
    listRpBinary = np.array([rp.getRpBinaryFromListDot(listDot) for listDot in listDotOfRp])
    label, info = getLabelAndInfo(recordName, type)
    return listRpBinary, label, info


def loadRpByCluster(numberOfCluster, indexCluster, listRecordNames=None, type='train'):
    if listRecordNames is None:
        listRecordNames = config.NAME_OF_RECORD
    listRpBinary = []
    listLabel = []
    listInfo = []
    for recordName in listRecordNames:
        print('_____ read rp from record ', recordName)
        # startTime = datetime.datetime.now()
        # print('startTime: ', startTime)

        dataFile = config.getFileSaveRp(recordName, type)
        allRpDot = np.load(dataFile, allow_pickle=True)
        label, info = getLabelAndInfo(recordName, type)

        for i in range(indexCluster, len(allRpDot), numberOfCluster):
            # Lấy các i thỏa mãn dk: chia cho numberOfCluster có số dư là indexCluster
            rpDot = allRpDot[i]
            rpBinary = rp.getRpBinaryFromListDot(rpDot)

            listRpBinary.append(rpBinary)
            listLabel.append(label[i])
            listInfo.append(info[i])
        # endTime = datetime.datetime.now()
        # print('duration load: ', endTime - start)
    return np.array(listRpBinary), np.array(listLabel), np.array(listInfo)


def loadRqa(recordName='a01', type='train'):
    # todo: use getFileName
    dataFile = config.getFileSaveRqa(recordName, type)
    listRqa = np.load(dataFile, allow_pickle=True)
    indexColOfRqaParam = config.RQA_DATA_USE_COL
    listRqa = listRqa[:, indexColOfRqaParam]
    label, info = getLabelAndInfo(recordName, type)
    return listRqa, label, info


def loadAllRqa(recordNameForTest, recordNameForTrain=None):
    allRecordNames = config.NAME_OF_RECORD
    if recordNameForTrain is None:
        recordNameForTrain = [recordName
                              for recordName in allRecordNames
                              if recordName not in recordNameForTest]
    print('record for train: ', recordNameForTrain)
    print('record for test: ', recordNameForTest)

    trainRqa = []
    trainLabel = []
    for recordName in recordNameForTrain:
        rqa, label, _ = loadRqa(recordName, 'train')
        trainRqa = np.append(trainRqa, rqa, axis=0) if len(trainRqa) > 0 else rqa
        trainLabel = np.append(trainLabel, label, axis=0) if len(trainLabel) > 0 else label

    testRqa = []
    testLabel = []
    for recordName in recordNameForTest:
        rqa, label, _ = loadRqa(recordName, 'test')
        testRqa = np.append(testRqa, rqa, axis=0) if len(testRqa) > 0 else rqa
        testLabel = np.append(testLabel, label, axis=0) if len(testLabel) > 0 else label
    return trainRqa, trainLabel, testRqa, testLabel


def getLabel(labels):
    """
    :param labels:
    :return: APNEA_LABEL if number of apnea > 70%, else return NORMAL_LABEL
    """
    unique, counts = np.unique(labels, return_counts=True)
    dictUnique = dict(zip(unique, counts))
    apnea = config.APNEA_LABEL
    normal = config.NORMAL_LABEL
    minPercentApnea = config.MIN_PERCENT_APNEA
    if apnea in dictUnique.keys() and dictUnique[apnea] >= minPercentApnea * len(labels):
        return apnea
    return normal


if __name__ == '__main__':
    print('MyUtil run main')
    recordNames = config.NAME_OF_RECORD
    recordNameForTest = ['a03', 'a05', 'a13', 'a16', 'a19']
    trainRqa, trainLabel, testRqa, testLabel = loadAllRqa(recordNameForTest)
    print('done')

    print('trainRqa: ', trainRqa.shape)
    # print(trainRqa[-10:])
    print('trainLabel: ', trainLabel.shape)
    print('testRqa: ', testRqa.shape)
    # print(testRqa[-10:])
    print('testLabel: ', testLabel.shape)

    unique, count = np.unique(trainLabel, return_counts=True)
    print('trainLabel unique: ', dict(zip(unique, count)))
    unique, count = np.unique(testLabel, return_counts=True)
    print('testLabel unique: ', dict(zip(unique, count)))

    # data, label, info = readRpBinary(recordNames[2])
    # print('data:', data.shape)
    # print('label:', label.shape)
    # print('info:', info.shape)

    # from src import RecurrentPlot as rp
    #
    # createFolder(config.PATH_RP_TRAIN_NORMAL)
    # createFolder(config.PATH_RP_TRAIN_APNEA)
    # for iData in [0, 5, 9]:
    #     print('iData: ', iData)
    #     listRpBinary, label, info = readRpBinary(recordNames[iData])
    #
    #     for i in range(10, 15, 1):
    #         rpBinary = listRpBinary[i]
    #         thisLabel = label[i]
    #         title = 'N-' if thisLabel == config.NORMAL_LABEL else 'A-'
    #         title += 'record-{}.start-{}.end-{}'.format(iData, info[i][1], info[i][2])
    #         folderSave = config.PATH_RP_TRAIN_NORMAL if thisLabel == config.NORMAL_LABEL \
    #             else config.PATH_RP_TRAIN_APNEA
    #         pathSaveImage = folderSave + title + config.IMG_SUFFIX if config.IS_SAVE_RP_IMAGE else None
    #         rp.crossRecurrencePlots(title, dataMatrixBinary=rpBinary, myTitle=title, showPlot=True,
    #                                 pathSaveFigure=pathSaveImage)
    #
    # print('finish code')
