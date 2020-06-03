# from numba import jit, cuda
import numpy as np
from tqdm import tqdm
from tqdm import trange
from threading import Thread
import multiprocessing

from src import config
from src import MyUtil as myUtil
from src import RecurrentPlot as rp
from src import RecurrenceQuantificationAnalysis as rqa

# ============================== Load Config =================================#
trainDataFile = config.FILE_RRI_NPY
trainLabelFile = config.FILE_RRI_LABEL

winSize = config.RR_PER_RECURRENCE_PLOTS
winStep = config.WIN_STEP_SIZE
dim = config.DIMENSION
tau = config.TAU
e = config.EPSILON
disNorm = config.DISTANCE_NORM
dotRate = config.DOT_RATE
isFixedEpsilon = config.IS_FIXED_EPSILON

myLambda = config.MY_LAMBDA


# ============================================================================#


def loadRri(dataFile, labelFile):
    print('load rri . . .')
    trainDataOrigin = np.load(dataFile, allow_pickle=True)
    labelOrigin = np.load(labelFile, allow_pickle=True)

    rriData = []  # [[record1], [record2], ...]
    label = []  # [[listLabels1], [listLabels2], ...]
    indexWhereStartMinute = []
    # ===> rriData[i][j] has label[i][j]

    curRecordIndex = None
    curContainer = []
    curListLabel = []
    curListIndexStartMinute = []
    for i, originData in tqdm(enumerate(trainDataOrigin)):
        rri = originData[0]
        recordIndex = originData[1]
        if recordIndex == curRecordIndex:
            curListIndexStartMinute.append(len(curContainer))
            curContainer = np.append(curContainer, np.array(rri))
            curListLabel += [labelOrigin[i]] * len(rri)
        else:
            # curRecordIndex is not None when end 1 record
            # curRecordIndex is None when start load
            if curRecordIndex is not None:
                # curRecordIndex is not None
                rriData.append(curContainer)
                label.append(curListLabel)
                indexWhereStartMinute.append(curListIndexStartMinute)
            # curRecordIndex is None
            curRecordIndex = recordIndex
            curContainer = rri
            curListLabel = [labelOrigin[i]] * len(rri)
            curListIndexStartMinute = [0]
            print("\ncurRecordIndex: ", curRecordIndex)

    # push last record
    rriData.append(curContainer)
    label.append(curListLabel)
    indexWhereStartMinute.append(curListIndexStartMinute)
    return rriData, label, indexWhereStartMinute


def makeRpAndRqa(timeSeries, rpContainer=None, rqaContainer=None):
    # Make rp, rqa from timeSeries
    binaryMatrix = rp.makeRpMatrix(timeSeries, dim, tau, e, disNorm, isFixedEpsilon=isFixedEpsilon,
                                   dotRate=dotRate)
    if config.IS_SAVE_RP_DOT:
        dotOfBinary = rp.getDotOfRpBinary(binaryMatrix)
        if rpContainer is not None:
            rpContainer.append(dotOfBinary)
    if config.IS_SAVE_RQA:
        thisRqa = rqa.rqaCalculate(binaryMatrix, lambd=myLambda)
        if rqaContainer is not None:
            rqaContainer.append(thisRqa)
    if config.IS_SAVE_RP_IMAGE or config.IS_SHOW_RP:
        if thisLabel == config.NORMAL_LABEL:
            folderSave = config.PATH_RP_TRAIN_NORMAL
            title = 'N-'
        else:
            folderSave = config.PATH_RP_TRAIN_APNEA
            title = 'A-'
        title += 'record-{}.start-{}.end-{}'.format(recordIndex, start, end)
        pathSaveImage = folderSave + title + config.IMG_SUFFIX if config.IS_SAVE_RP_IMAGE else None

        rp.crossRecurrencePlots(windowTitle=title, dataMatrixBinary=binaryMatrix, myTitle=title,
                                pathSaveFigure=pathSaveImage, showPlot=config.IS_SHOW_RP)


def saveData(recordName, type='train', labelContainer=None, infoContainer=None, rpContainer=None,
             rqaContainer=None):
    if config.IS_SAVE_LABEL_AND_INFO:
        fileSaveLabel = config.getFileSaveLabel(recordName, type)
        fileSaveInfo = config.getFileSaveInfo(recordName, type)
        print('save ', fileSaveLabel)
        if labelContainer is not None:
            np.save(fileSaveLabel, labelContainer)
        print('save ', fileSaveInfo)
        if infoContainer is not None:
            np.save(fileSaveInfo, infoContainer)

    if config.IS_SAVE_RP_DOT:
        fileSaveRp = config.getFileSaveRp(recordName, type)
        print('save ', fileSaveRp)
        if rpContainer is not None:
            np.save(fileSaveRp, rpContainer)

    if config.IS_SAVE_RQA:
        fileSaveRqa = config.getFileSaveRqa(recordName, type)
        print('save ', fileSaveRqa)
        if rqaContainer is not None:
            np.save(fileSaveRqa, rqaContainer)


def makeTrainData(allData, allLabel, listIndexOfRecord):
    # # ============================= MAKE TRAIN DATA =========================
    for recordIndex in listIndexOfRecord:
        rriData = allData[recordIndex]
        # if i_data > 1:
        #     break
        print('make train recordIndex: {}, len of record: {}, len of Label: {}'
              .format(recordIndex, len(rriData), len(allLabel[recordIndex])))
        if len(rriData) > winSize:
            rpOfThisRecord = []
            rqaOfThisRecord = []
            labelOfThisRecord = []
            infoOfThisRecord = []

            recordName = config.NAME_OF_RECORD[recordIndex]
            pbar = tqdm(range(0, len(rriData) - winSize, winStep))
            for start in pbar:
                pbar.set_description('Record index: {}'.format(recordIndex))
                end = start + winSize
                # end = start + 100
                timeSeries = rriData[start:end]
                # timeSeries = convertSetNumber(timeSeries)
                makeRpAndRqa(timeSeries, rpOfThisRecord, rqaOfThisRecord)
                # ================= get start of last minute ============================
                startMinute = end
                sumSec = rriData[end]
                while sumSec + rriData[startMinute - 1] < config.MINUTE + config.BIAS_MINUTE:
                    startMinute -= 1
                    sumSec += rriData[startMinute]
                # ======================================================================
                thisLabel = myUtil.getLabel(allLabel[recordIndex][startMinute:end])
                labelOfThisRecord.append(thisLabel)
                infoOfThisRecord.append([recordIndex, start, end])

            # ------------------------- done for one record -------------------------
            print('done make train for ', recordName)
            saveData(recordName, 'train', labelOfThisRecord, infoOfThisRecord, rpOfThisRecord, rqaOfThisRecord)
        else:
            print(" len of data < winSize({})".format(winSize))


def makeTestData(allData, allLabel, allIndexStartMinute, listIndexOfRecord):
    # ============================= MAKE TEST DATA =========================
    for recordIndex in listIndexOfRecord:
        rriData = allData[recordIndex]
        print('make Test recordIndex: {}, len of record: {}, len of Label: {}'
              .format(recordIndex, len(rriData), len(allLabel[recordIndex])))
        rpOfThisRecord = []
        rqaOfThisRecord = []
        labelOfThisRecord = []
        infoOfThisRecord = []
        listIndexStartMinute = allIndexStartMinute[recordIndex]

        recordName = config.NAME_OF_RECORD[recordIndex]
        pbar = tqdm(range(len(listIndexStartMinute) - 1, 0, -1))
        for iMinute in pbar:
            pbar.set_description('makeTest Record index: {}'.format(recordIndex))
            end = listIndexStartMinute[iMinute]
            start = end - winSize
            if start < 0:
                break
            timeSeries = rriData[start:end]
            # ========= Check max rri =========
            if np.max(timeSeries) > config.MAX_RRI_BY_SEC:
                print('rri out of range index start: {}, rri: {}'.format(start, np.max(timeSeries)))
                continue
            # timeSeries = convertSetNumber(timeSeries)
            makeRpAndRqa(timeSeries, rpOfThisRecord, rqaOfThisRecord)
            # ================= get Label ==========================================
            startLastMinute = listIndexStartMinute[iMinute - 1]
            listLabelInLastMinute = allLabel[recordIndex][startLastMinute:end]
            if len(np.unique(listLabelInLastMinute)) != 1:
                print('error label test: ', recordIndex, start, end, listLabelInLastMinute)
            thisLabel = listLabelInLastMinute[1]
            # ======================================================================
            labelOfThisRecord.append(thisLabel)
            infoOfThisRecord.append([recordIndex, start, end])

            if start % 50 == 0:
                saveData(recordName, 'test', labelOfThisRecord, infoOfThisRecord, rpOfThisRecord, rqaOfThisRecord)
                print('//-------------- done save file {} at index: {} --------------//'.format(recordName, start))
        # ------------------------- done for one record -------------------------
        print('done make rp for ', recordName)
        saveData(recordName, 'test', labelOfThisRecord, infoOfThisRecord, rpOfThisRecord, rqaOfThisRecord)


if __name__ == '__main__':
    print("Preprocess2.py run main")

    allData, allLabel, allIndexStartMinute = loadRri(trainDataFile, trainLabelFile)
    numRecordLoaded = len(allData)
    if numRecordLoaded != len(config.NAME_OF_RECORD):
        print('len error: \n'
              'num record loaded: {}\n'
              'num record name:   {}\n'
              .format(numRecordLoaded, len(config.NAME_OF_RECORD)))
    else:
        for iRecord in range(len(allData)):
            print(iRecord, ' >3: ', np.count_nonzero(allData[iRecord] > 3))

        numberOfTrainRecord = config.NUMBER_OF_TRAIN_RECORD

        myUtil.createFolder(config.PATH_RP_TRAIN)
        myUtil.createFolder(config.PATH_RP_TEST)
        if config.IS_SAVE_RP_IMAGE:
            myUtil.createFolder(config.PATH_RP_TRAIN_NORMAL)
            myUtil.createFolder(config.PATH_RP_TRAIN_APNEA)

        numTrain = config.NUMBER_OF_TRAIN_RECORD

        numProcess = 4
        # ============================= MAKE TEST DATA --MULTIPLE THREAD-- =========================
        myProcesses = []


        # @jit(target='cuda')


        for soDu in range(0, numProcess):
            listIndexOfRecord = []
            for index in range(numRecordLoaded):
                if index % numProcess == soDu:
                    listIndexOfRecord.append(index)
            myProc = multiprocessing.Process(target=makeTestData,
                                             args=(allData, allLabel, allIndexStartMinute, listIndexOfRecord))
            myProcesses.append(myProc)
        for i, myProcess in enumerate(myProcesses):
            myProcess.start()
        for i, myProcess in enumerate(myProcesses):
            myProcess.join()

        # ============================= MAKE TRAIN DATA WITH --MULTIPLE THREAD-- =========================
        # myProcesses = []
        # for soDu in range(0, numProcess):
        #     listIndexOfRecord = []
        #     for index in range(numRecordLoaded):
        #         if index % numProcess == soDu:
        #             listIndexOfRecord.append(index)
        #     myProc = multiprocessing.Process(target=makeTrainData,
        #                                      args=(allData, allLabel, listIndexOfRecord))
        #     myProcesses.append(myProc)
        # for i, myProcess in enumerate(myProcesses):
        #     myProcess.start()
        # for i, myProcess in enumerate(myProcesses):
        #     myProcess.join()
        #
        # print('done make train')
        # import datetime
        #
        # currentDT = datetime.datetime.now()
        # print(str(currentDT))

        # # ============================= MAKE TRAIN DATA =========================
        # for recordIndex in range(0, numTrain):
        #     rriData = allData[recordIndex]
        #     # if i_data > 1:
        #     #     break
        #     print('make train recordIndex: {}, len of record: {}, len of Label: {}'
        #           .format(recordIndex, len(rriData), len(allLabel[recordIndex])))
        #     if len(rriData) > winSize:
        #         rpOfThisRecord = []
        #         rqaOfThisRecord = []
        #         labelOfThisRecord = []
        #         infoOfThisRecord = []
        #         for start in tqdm(range(0, len(rriData) - winSize, winStep)):
        #             end = start + winSize
        #             # end = start + 100
        #             timeSeries = rriData[start:end]
        #             # timeSeries = convertSetNumber(timeSeries)
        #             makeRpAndRqa(timeSeries, rpOfThisRecord, rqaOfThisRecord)
        #             # ================= get start of last minute ============================
        #             startMinute = end
        #             sumSec = rriData[end]
        #             while sumSec + rriData[startMinute - 1] > config.MINUTE + config.BIAS_MINUTE:
        #                 startMinute -= 1
        #                 sumSec += rriData[startMinute]
        #             # ======================================================================
        #             thisLabel = myUtil.getLabel(allLabel[recordIndex][startMinute:end])
        #             labelOfThisRecord.append(thisLabel)
        #             infoOfThisRecord.append([recordIndex, start, end])
        #         # ------------------------- done for one record -------------------------
        #         recordName = config.NAME_OF_RECORD[recordIndex]
        #         print('done make rp for ', recordName)
        #         saveData(recordName, 'train', labelOfThisRecord, infoOfThisRecord, rpOfThisRecord, rqaOfThisRecord)
        #     else:
        #         print(" len of data < winSize({})".format(winSize))

        # # ============================= MAKE TEST DATA =========================
        # # for recordIndex in range(numTrain, len(allData)):
        # for recordIndex in range(0, numTrain):
        #     rriData = allData[recordIndex]
        #     print('make Test recordIndex: {}, len of record: {}, len of Label: {}'
        #           .format(recordIndex, len(rriData), len(allLabel[recordIndex])))
        #     rpOfThisRecord = []
        #     rqaOfThisRecord = []
        #     labelOfThisRecord = []
        #     infoOfThisRecord = []
        #     listIndexStartMinute = allIndexStartMinute[recordIndex]
        #     for iMinute in tqdm(range(len(listIndexStartMinute) - 1, 0, -1)):
        #         end = listIndexStartMinute[iMinute]
        #         start = end - winSize
        #         if start < 0:
        #             break
        #         timeSeries = rriData[start:end]
        #         # ========= Check max rri =========
        #         if np.max(timeSeries) > config.MAX_RRI_BY_SEC:
        #             print('rri out of range index start: {}, rri: {}'.format(start, np.max(timeSeries)))
        #             continue
        #         # timeSeries = convertSetNumber(timeSeries)
        #         makeRpAndRqa(timeSeries, rpOfThisRecord, rqaOfThisRecord)
        #         # ================= get Label ==========================================
        #         startLastMinute = listIndexStartMinute[iMinute - 1]
        #         listLabelInLastMinute = allLabel[recordIndex][startLastMinute:end]
        #         if len(np.unique(listLabelInLastMinute)) != 1:
        #             print('error label test: ', recordIndex, start, end, listLabelInLastMinute)
        #         thisLabel = listLabelInLastMinute[1]
        #         # ======================================================================
        #         labelOfThisRecord.append(thisLabel)
        #         infoOfThisRecord.append([recordIndex, start, end])
        #     # ------------------------- done for one record -------------------------
        #     recordName = config.NAME_OF_RECORD[recordIndex]
        #     print('done make rp for ', recordName)
        #     saveData(recordName, 'test', labelOfThisRecord, infoOfThisRecord, rpOfThisRecord, rqaOfThisRecord)
