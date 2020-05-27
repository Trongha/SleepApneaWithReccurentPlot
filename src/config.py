# ==================================== turn On - Off ====================================
IS_SAVE_RP_DOT = True
IS_SAVE_LABEL_AND_INFO = True
IS_SAVE_RQA = True

IS_SAVE_RP_IMAGE = False
IS_SHOW_RP = False
# =======================================================================================


res = '../res/'
MINUTE = 60

# =============================== config for Data preprocress ===========================
FOLDER_SAVE_ORIGIN_DATA = res + 'origin/'
NUMBER_OF_TRAIN_RECORD = 20
PATH_RRI = '../res/rri/'
FILE_RRI_NPY = PATH_RRI + 'rri.npy'  # [[[listRri], recodeIndex], [], ...]
FILE_RRI_LABEL = PATH_RRI + 'rri.lb.npy'

NORMAL_LABEL = 0
APNEA_LABEL = 1
NONE_LABEL = 2

NAME_OF_RECORD = ['a01', 'a02', 'a03', 'a04', 'a05',
                  'a06', 'a07', 'a08', 'a09', 'a10',
                  'a11', 'a12', 'a13', 'a14', 'a15',
                  'a16', 'a17', 'a18', 'a19',
                  'b01', 'b02', 'b03', 'b04',
                  'c01', 'c02', 'c03', 'c04', 'c05',
                  'c06', 'c07', 'c08', 'c09',
                  'a20', 'b05', 'c10'
                  ]
# =======================================================================================


# ==================================== config for RP ====================================
MIN_PERCENT_APNEA = 0.7
RR_PER_RECURRENCE_PLOTS = 450
WIN_STEP_SIZE = 50
DIMENSION = 6
TAU = 10
# ===> 1 Rp co 25 statePhase
EPSILON = 0.5
DISTANCE_NORM = 2
IS_FIXED_EPSILON = False
DOT_RATE = 0.2

PATH_RP = res + 'rp/'
PATH_RP_TRAIN = PATH_RP + 'train/'
PATH_RP_TEST = PATH_RP + 'test/'

PATH_RP_TRAIN_NORMAL = PATH_RP_TRAIN + 'normalImage/'
PATH_RP_TRAIN_APNEA = PATH_RP_TRAIN + 'apneaImage/'

FILE_RP_TRAIN_NORMAL = PATH_RP + 'rp_train_normal.npy'  # [[rp, label], [rp, label]]
FILE_RP_TRAIN_APNEA = PATH_RP + 'rp_train_apnea.npy'

IMG_SUFFIX = '.png'
RP_SUFFIX = '.rp'
INFO_SUFFIX = '.inf'
LABEL_SUFFIX = '.lb'
RQA_SUFFIX = '.rqa'
# ========================================================================================


# ==================================== config for RQA ====================================
MY_LAMBDA = 80
PATH_RQA = res + 'rqa/'
PATH_RQA_TRAIN = PATH_RQA + 'train/'
# =======================================================================================


def getFileTxtRri(recordName):
    return PATH_RRI + recordName + '.rri.txt'


def getFileSaveRp(recordName, type='train'):
    folder = PATH_RP_TRAIN if type == 'train' else PATH_RP_TEST
    return folder + recordName + '.rp.npy'


def getFileSaveLabel(recordName, type='train'):
    # format of file: [[i_data, start, end], [i_data, start, end], ...]
    folder = PATH_RP_TRAIN if type == 'train' else PATH_RP_TEST
    return folder + recordName + '.lb.npy'


def getFileSaveInfo(recordName, type='train'):
    folder = PATH_RP_TRAIN if type == 'train' else PATH_RP_TEST
    return folder + recordName + '.inf.npy'


def getFileSaveRqa(recordName, type='train'):
    folder = PATH_RP_TRAIN if type == 'train' else PATH_RP_TEST
    return folder + recordName + '.rqa.npy'
