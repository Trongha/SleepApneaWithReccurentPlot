import os
from typing import Union, Iterable

import numpy as np
from numpy.core._multiarray_umath import ndarray

from src import config as config


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)




def getLabelAndInfo(recordName='a01'):
    labelFile = config.getFileSaveLabel(recordName)
    infoFile = config.getFileSaveInfo(recordName)
    label = np.load(labelFile, allow_pickle=True)
    info = np.load(infoFile, allow_pickle=True)
    return label, info


def readRpBinary(recordName='a01'):
    # todo: use getFileName
    dataFile = config.getFileSaveRp(recordName)
    listRpBinary = np.load(dataFile, allow_pickle=True)
    label, info = getLabelAndInfo(recordName)
    return listRpBinary, label, info


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
        data, label, info = readRpBinary(recordNames[iData])
        start = 0
        while (label[start] != config.APNEA_LABEL):
            start += 1

        for i in range(start, start + 5, 1):
            listDot = data[i]
            thisLabel = label[i]
            title = 'N-' if thisLabel == config.NORMAL_LABEL else 'A-'
            title += 'record-{}.start-{}.end-{}'.format(iData, info[i][1], info[i][2])
            folderSave = config.PATH_RP_TRAIN_NORMAL if thisLabel == config.NORMAL_LABEL \
                else config.PATH_RP_TRAIN_APNEA
            pathSaveImage = folderSave + title + config.IMG_SUFFIX if config.IS_SAVE_RP_IMAGE else None
            rpBinary = rp.getRpBinaryFromListDot(listDot)
            rp.crossRecurrencePlots(title, rpBinary, myTitle=title, showPlot=True, pathSaveFigure=pathSaveImage)
        for i in range(10, 15, 1):
            listDot = data[i]
            thisLabel = label[i]
            title = 'N-' if thisLabel == config.NORMAL_LABEL else 'A-'
            title += 'record-{}.start-{}.end-{}'.format(iData, info[i][1], info[i][2])
            folderSave = config.PATH_RP_TRAIN_NORMAL if thisLabel == config.NORMAL_LABEL \
                else config.PATH_RP_TRAIN_APNEA
            pathSaveImage = folderSave + title + config.IMG_SUFFIX if config.IS_SAVE_RP_IMAGE else None
            rpBinary = rp.getRpBinaryFromListDot(listDot)
            rp.crossRecurrencePlots(title, dataMatrixBinary=rpBinary, myTitle=title, showPlot=True,
                                    pathSaveFigure=pathSaveImage)

    print('finish code')
