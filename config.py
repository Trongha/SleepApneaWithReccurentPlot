res = '../res/'
FOLDER_SAVE_ORIGIN_DATA = res + 'origin/'
NUMBER_OF_TRAIN = 20

# =============================== config for Data preprocress ===========================
NORMAL_LABEL = 0
APNEA_LABEL = 1
NONE_LABEL = 2
FOLDER_SAVE_PREPROCESS = res + 'dataPreProcess/'
RRI_DATA_FILE = FOLDER_SAVE_PREPROCESS + 'my_train_input.npy'
RRI_LABEL_FILE = FOLDER_SAVE_PREPROCESS + 'my_train_label.npy'

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
WIN_STEP_SIZE = 10
DIMENSION = 6
TAU = 10
# ===> 1 Rp co 25 statePhase
EPSILON = 0.5
DISTANCE_NORM = 2
IS_FIXED_EPSILON = False
DOT_RATE = 0.2

IS_SAVE_RP_IMAGE = False
IS_SAVE_RP_BINARY = False
IS_SHOW_RP = False

PATH_RP = res + 'rp/'
PATH_RP_TRAIN = PATH_RP + 'train/'

PATH_RP_TRAIN_NORMAL = PATH_RP_TRAIN + 'normal/'
PATH_RP_TRAIN_APNEA = PATH_RP_TRAIN + 'apnea/'

FILE_RP_TRAIN_NORMAL = PATH_RP + 'rp_train_normal.npy'  # [[rp, label], [rp, label]]
FILE_RP_TRAIN_APNEA = PATH_RP + 'rp_train_apnea.npy'

IMG_SUFFIX = '.png'
RP_SUFFIX = '.rp'
INFO_SUFFIX = '.inf'
LABEL_SUFFIX = '.lb'
RQA_SUFFIX = '.rqa'


def getFileSaveRp(recordName):
    return PATH_RP_TRAIN + recordName + '.rp.npy'


def getFileSaveLabel(recordName):
    return PATH_RP_TRAIN + recordName + '.lb.npy'


def getFileSaveInfo(recordName):
    return PATH_RP_TRAIN + recordName + '.inf.npy'


def getFileSaveRqa(recordName):
    return PATH_RP_TRAIN + recordName + '.rqa.npy'
# ========================================================================================


# ==================================== config for RQA ====================================
IS_SAVE_RQA = True
MY_LAMBDA = 80
PATH_RQA = res + 'rqa/'
PATH_RQA_TRAIN = PATH_RQA + 'train/'

# =======================================================================================
