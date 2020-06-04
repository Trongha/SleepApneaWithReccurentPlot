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
    # todo: use getFileName
    if listRecordNames is None:
        listRecordNames = config.NAME_OF_RECORD
    listRpBinary = None
    listLabel = None
    listInfo = None
    for recordName in listRecordNames:
        print('_____ read rp from record ', recordName)
        # startTime = datetime.datetime.now()
        # print('startTime: ', startTime)

        dataFile = config.getFileSaveRp(recordName, type)
        allRpDot = np.load(dataFile, allow_pickle=True)
        label, info = getLabelAndInfo(recordName, type)

        for i in range(indexCluster, len(allRpDot), numberOfCluster):
            # Lấy các i thỏa mãn chia cho numberOfCluster có số dư là indexCluster
            rpDot = allRpDot[i]
            rpBinary = rp.getRpBinaryFromListDot(rpDot)
            if listRpBinary is not None:
                listRpBinary = np.append(listRpBinary, [rpBinary], axis=0)
                listLabel = np.append(listLabel, [label[i]], axis=0)
                listInfo = np.append(listInfo, [info[i]], axis=0)
            else:
                listRpBinary = np.array([rpBinary])
                listLabel = np.array([label[i]])
                listInfo = np.array([info[i]])
        # endTime = datetime.datetime.now()
        # print('duration load: ', endTime - statr)
    return listRpBinary, listLabel, listInfo


def loadRqa(recordName='a01', type='train'):
    # todo: use getFileName
    dataFile = config.getFileSaveRqa(recordName, type)
    listRqa = np.load(dataFile, allow_pickle=True)
    indexColOfRqaParam = [1, 2, 4, 5, 6, 7, 9, 10, 11]
    listRqa = listRqa[:, indexColOfRqaParam]
    label, info = getLabelAndInfo(recordName, type)
    return listRqa, label, info


def getLabel(labels):
    '''
    :param labels:
    :return: APNEA_LABEL if number of apnea > 50%, else return NORMAL_LABEL
    '''
    unique, counts = np.unique(labels, return_counts=True)
    dictUnique = dict(zip(unique, counts))
    apnea = config.APNEA_LABEL
    normal = config.NORMAL_LABEL
    minPercentApnea = config.MIN_PERCENT_APNEA
    if apnea in dictUnique.keys() and dictUnique[apnea] >= minPercentApnea * len(labels):
        # print('label : apnea')
        return apnea
    return normal


if __name__ == '__main__':
    print('MyUtil run main')
    recordNames = config.NAME_OF_RECORD

    # data, label, info = readRpBinary(recordNames[2])
    # print('data:', data.shape)
    # print('label:', label.shape)
    # print('info:', info.shape)

    from src import RecurrentPlot as rp

    createFolder(config.PATH_RP_TRAIN_NORMAL)
    createFolder(config.PATH_RP_TRAIN_APNEA)
    for iData in [0, 5, 9]:
        print('iData: ', iData)
        listRpBinary, label, info = readRpBinary(recordNames[iData])

        for i in range(10, 15, 1):
            rpBinary = listRpBinary[i]
            thisLabel = label[i]
            title = 'N-' if thisLabel == config.NORMAL_LABEL else 'A-'
            title += 'record-{}.start-{}.end-{}'.format(iData, info[i][1], info[i][2])
            folderSave = config.PATH_RP_TRAIN_NORMAL if thisLabel == config.NORMAL_LABEL \
                else config.PATH_RP_TRAIN_APNEA
            pathSaveImage = folderSave + title + config.IMG_SUFFIX if config.IS_SAVE_RP_IMAGE else None
            rp.crossRecurrencePlots(title, dataMatrixBinary=rpBinary, myTitle=title, showPlot=True,
                                    pathSaveFigure=pathSaveImage)

    print('finish code')
