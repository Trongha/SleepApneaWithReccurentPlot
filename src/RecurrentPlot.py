import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src import config
import src.RecurrenceQuantificationAnalysis as rqa
import src.MyUtil as myUtil

# res = '../res/train/'
res = '../output/'
trainDataFile = config.FILE_RRI_NPY
trainLabelFile = config.FILE_RRI_LABEL


def convert2StatePhase(timeSeries, dim, tau, returnType='array'):
    # v: vecto
    # dim là số chiều (số phần tử)
    # tau là bước nhảy
    # returnType: kiểu trả về
    #	array: trả về python array
    #	np.array: trả về mảng numpy
    if returnType == 'array':
        return [timeSeries[start: start + (dim - 1) * tau + 1: tau]
                for start in range(len(timeSeries) - (dim - 1) * tau)]
    if returnType == 'np.array':
        import numpy as np
        return np.array([timeSeries[start: start + (dim - 1) * tau + 1: tau]
                         for start in range(len(timeSeries) - (dim - 1) * tau)])


def makeRpMatrix(timeSeries, dim=5, tau=2, epsilon=0.09, distNorm=2, isFixedEpsilon=True, dotRate=0.2):
    # tách statephases
    statePhase = convert2StatePhase(timeSeries, dim, tau, 'np.array')

    from scipy.spatial.distance import cdist
    # rDist là ma trận khoảng cách
    # cdist là hàm trong scipy.spatial.distance.cdist
    # minkowski là cách tính
    # p là norm
    # y là train: đánh số từ trên xuống dưới
    # x là test
    rDist = cdist(statePhase, statePhase, 'minkowski', p=distNorm)

    if isFixedEpsilon:
        rBinary = np.array((rDist < epsilon) + 0)
        return rBinary
    else:
        for i in range(rDist.shape[0]):
            row = rDist[i]
            rDist[i] = getBinaryByRow(row, dotRate)
        return rDist


def getBinaryByRow(row, dotRate=0.2):
    rowClone = np.array(row)
    # print('index: ', -int(dotRate * len(row)))
    row.sort()
    epsilon = row[int(dotRate * len(row))]
    return (rowClone < epsilon) + 0


# vẽ biểu đồ chấm từ mảng x và mảng y
def scatterGraph(windowTitle, dataX, dataY, dotSize=0, myTitle='scatterGraph', labelX='xxxxx', labelY='yyyyy'):
    f = plt.figure(windowTitle)
    plt.scatter(dataX, dataY, s=dotSize)
    plt.title(myTitle)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    return f


# vẽ biểu đồ crp từ ma trận 01
def crossRecurrencePlots(windowTitle, dataMatrixBinary, keyDot=1, dotSize=1, myTitle='prettyGirl', labelX='xxxxx',
                         labelY='yyyyy', showPlot=False, pathSaveFigure=None):
    dataX = []
    dataY = []
    hightOfData = len(dataMatrixBinary)

    # print("crossRecurrencePlots()_len: ", hightOfData)
    for y in range(hightOfData):
        for x in range(len(dataMatrixBinary[y])):
            if dataMatrixBinary[y][x] == keyDot:
                dataX.append(x)
                # append hight-y nếu muốn vẽ đồ thị đúng chiều như lưu trong ma trận
                dataY.append(hightOfData - y - 1)
            # vẽ trục y từ dưới lên
            # dataY.append(y);

    figure = scatterGraph(windowTitle, dataX, dataY, dotSize, myTitle, labelX, labelY)
    if showPlot:
        plt.show()
    if pathSaveFigure is not None:
        plt.savefig(pathSaveFigure, dpi=200)
    return figure


def convertSetNumber(Set, minOfSet=0, maxOfSet=0, newMinOfSet=0, newMaxOfSet=1):
    # Chuan hoa ve [0,1]
    if minOfSet == 0:
        minOfSet = min(Set)
    if maxOfSet == 0:
        maxOfSet = max(Set)

    # print("min: ", minOfSet)
    # print("max: ", maxOfSet)

    if maxOfSet == minOfSet:
        ratio = 0
    else:
        ratio = (newMaxOfSet - newMinOfSet) / (maxOfSet - minOfSet)
    return [((x - minOfSet) * ratio + newMinOfSet) for x in Set]


def getDotOfRpBinary(rpBinary, keyDot=1):
    dots = []
    for row in rpBinary:
        dots.append([i for i, x in enumerate(row) if x == keyDot])
    return dots


def getRpBinaryFromListDot(listDot, keyDot=1):
    l = len(listDot)
    rpBinary = np.zeros((l, l))
    for y, row in enumerate(listDot):
        for x in row:
            rpBinary[y][x] = keyDot
    return rpBinary


# ============================== read data =================================#
def readData(dataFile, labelFile):
    trainDataOrigin = np.load(dataFile, allow_pickle=True)
    labelOrigin = np.load(labelFile, allow_pickle=True)

    trainData = []
    label = []

    curMinute = None
    curContainer = []
    curListLabel = []
    sumRri = 0
    for i, originData in enumerate(trainDataOrigin):
        rri = originData[0]
        iMinute = originData[1]
        sumRri += np.sum(rri)
        print("sumRri: ", sumRri)
        if iMinute == curMinute:
            curContainer = np.append(curContainer, np.array(rri))
            curListLabel += [labelOrigin[i] for numberOf in range(len(rri))]
        else:
            if curMinute is not None:
                trainData.append(curContainer)
                label.append(curListLabel)

            curMinute = iMinute
            curContainer = rri
            curListLabel = [labelOrigin[i] for numberOf in range(len(rri))]
            print("curMinute: ", curMinute)
    trainData.append(curContainer)
    label.append(curListLabel)
    return trainData, label


if __name__ == '__main__':
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

    allData, allLabel = readData(trainDataFile, trainLabelFile)
    print('stop')
    trainData = allData[0:config.NUMBER_OF_TRAIN]
    trainLabel = allLabel[0:config.NUMBER_OF_TRAIN]

    print(len(trainData))
    print(len(trainLabel))

    myUtil.createFolder(config.PATH_RP_TRAIN)
    if config.IS_SAVE_RP_IMAGE:
        myUtil.createFolder(config.PATH_RP_TRAIN_NORMAL)
        myUtil.createFolder(config.PATH_RP_TRAIN_APNEA)

    rpNormal = []
    rpApnea = []
    # allRp = []
    for i_data, data in enumerate(trainData):
        # if i_data > 1:
        #     break
        print('len of item ', i_data, len(data), len(trainLabel[i_data]))
        if len(data) > winSize:
            allRp = []
            allRqa = []
            allLabel = []
            allInfo = []
            for start in tqdm(range(0, len(data) - winSize, winStep)):
                end = start + winSize
                # end = start + 100
                timeSeries = data[start:end]

                #================= get start of end minute ============================
                startMinute = end
                sumSec = data[end]
                while sumSec + data[startMinute - 1] < config.MINUTE:
                    startMinute -= 1
                    sumSec += data[startMinute]
                #======================================================================

                thisLabel = myUtil.getLabel(trainLabel[i_data][startMinute:end])
                # timeSeries = convertSetNumber(timeSeries)
                binaryMatrix = makeRpMatrix(timeSeries, dim, tau, e, disNorm, isFixedEpsilon=isFixedEpsilon,
                                            dotRate=dotRate)

                allLabel.append(thisLabel)
                allInfo.append([i_data, start, end])

                if config.IS_SAVE_RP_DOT:
                    dotOfBinary = getDotOfRpBinary(binaryMatrix)
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

                    x = crossRecurrencePlots(windowTitle=title, dataMatrixBinary=binaryMatrix, myTitle=title,
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

    # np.save(config.FILE_RP_TRAIN_NORMAL, rpNormal)

    # np.save(config.FILE_RP_TRAIN_APNEA, rpApnea)
