import numpy as np
from src import config as config
from src import MyUtil as myUtil
from tqdm import tqdm
from src import RecurrentPlot as rp
from src import RecurrenceQuantificationAnalysis as rqa

print("RecurrentPlot.py run main")
# ============================== Load Config =================================#
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

allData, allLabel, indexStartMinute = myUtil.loadRri(config.FILE_RRI_NPY, config.FILE_RRI_LABEL)

testData = allData[-config.NUMBER_OF_TRAIN_RECORD:]
testLabel = allLabel[-config.NUMBER_OF_TRAIN_RECORD:]
listIndexStartMinute = indexStartMinute[-config.NUMBER_OF_TRAIN_RECORD:]

for recordIndex, rriData in enumerate(testData):
    print('recordIndex: ', recordIndex)
    listStartIndex = listIndexStartMinute[recordIndex]
    for iMinute in tqdm(range(len(listStartIndex) - 1, config.RR_PER_RECURRENCE_PLOTS, -1)):
        endIndex = listStartIndex[iMinute]
        startIndex = endIndex - config.RR_PER_RECURRENCE_PLOTS
        if startIndex < 0:
            break
        timeSeries = rriData[startIndex:endIndex]


        # print(iMinute, 'sum rri: ', np.sum(rriData[startIndex:end]))

print(len(testData))
print(len(testLabel))

myUtil.createFolder(config.PATH_RP_TRAIN)
if config.IS_SAVE_RP_IMAGE:
    myUtil.createFolder(config.PATH_RP_TRAIN_NORMAL)
    myUtil.createFolder(config.PATH_RP_TRAIN_APNEA)

rpNormal = []
rpApnea = []
# allRp = []
for i_data, data in enumerate(testData):
    # if i_data > 1:
    #     break
    print('len of item ', i_data, len(data), len(testLabel[i_data]))
    if len(data) > winSize:
        allRp = []
        allRqa = []
        allLabel = []
        allInfo = []
        for start in tqdm(range(0, len(data) - winSize, winStep)):
            end = start + winSize
            # end = start + 100
            timeSeries = data[start:end]

            # ================= get start of end minute ============================
            startMinute = end
            sumSec = data[end]
            while sumSec + data[startMinute - 1] < config.MINUTE:
                startMinute -= 1
                sumSec += data[startMinute]
            # ======================================================================

            thisLabel = myUtil.getLabel(testLabel[i_data][startMinute:end])
            # timeSeries = convertSetNumber(timeSeries)

            binaryMatrix = rp.makeRpMatrix(timeSeries, dim, tau, e, disNorm, isFixedEpsilon=isFixedEpsilon,
                                           dotRate=dotRate)

            allLabel.append(thisLabel)
            allInfo.append([i_data, start, end])

            if config.IS_SAVE_RP_DOT:
                dotOfBinary = rp.getDotOfRpBinary(binaryMatrix)
                allRp.append(dotOfBinary)

            if config.IS_SAVE_RQA:
                thisRqa = rqa.rqaCalculate(binaryMatrix, lambd=myLambda)
                allRqa.append(thisRqa)

            if config.IS_SAVE_RP_IMAGE or config.IS_SHOW_RP:
                if thisLabel == config.NORMAL_LABEL:
                    folderSave = config.PATH_RP_TRAIN_NORMAL
                    title = 'N-'
                else:
                    folderSave = config.PATH_RP_TRAIN_APNEA
                    title = 'A-'

                title += 'record-{}.start-{}.end-{}'.format(i_data, start, end)
                pathSaveImage = folderSave + title + config.IMG_SUFFIX if config.IS_SAVE_RP_IMAGE else None

                x = rp.crossRecurrencePlots(windowTitle=title, dataMatrixBinary=binaryMatrix, myTitle=title,
                                            pathSaveFigure=pathSaveImage, showPlot=config.IS_SHOW_RP)
                # plt.show()

        # ------------------------- done for one record -------------------------
        recordName = config.NAME_OF_RECORD[i_data]
        print('done make rp for ', recordName)

        if config.IS_SAVE_LABEL_AND_INFO:
            fileSaveLabel = config.getFileSaveLabel(recordName)
            fileSaveInfo = config.getFileSaveInfo(recordName)
            print('save ', fileSaveLabel)
            np.save(fileSaveLabel, allLabel)
            print('save ', fileSaveInfo)
            np.save(fileSaveInfo, allInfo)

        if config.IS_SAVE_RP_DOT:
            fileSaveRp = config.getFileSaveRp(recordName)
            print('save ', fileSaveRp)
            np.save(fileSaveRp, allRp)

        if config.IS_SAVE_RQA:
            fileSaveRqa = config.getFileSaveRqa(recordName)
            print('save ', fileSaveRqa)
            np.save(fileSaveRqa, allRqa)
    else:
        print(" len of data < winSize({})".format(winSize))

dataFile = config.PATH_RQA + 'rqa.npy'
infoFile = config.PATH_RQA + 'info.npy'
rqa = np.load(dataFile, allow_pickle=True)
info = np.load(infoFile, allow_pickle=True)
labels = [i[3] for i in info]
for i, label in enumerate(labels):
    if label == config.APNEA_LABEL:
        print('apnea ', i, label)
print('stop')

# print(a, b)
# print(np.append(a, b))
