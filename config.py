res = '../res/'
FOLDER_SAVE_ORIGIN_DATA = res + 'origin/'

# =============================== config for Data preprocress ===========================
NORMAL_LABEL = 0
APNEA_LABEL = 1
NONE_LABEL = 2
FOLDER_SAVE_PREPROCESS = res + 'dataPreProcess/'
RRI_DATA_FILE = FOLDER_SAVE_PREPROCESS + 'my_train_input.npy'
RRI_LABEL_FILE = FOLDER_SAVE_PREPROCESS + 'my_train_label.npy'
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
IS_SAVE_RP_BINARY = True
IS_SHOW_RP = True

FOLDER_SAVE_RP = res + 'rp/'
SAVE_NORMAL_RP = FOLDER_SAVE_RP + 'normal/'
SAVE_APNEA_RP = FOLDER_SAVE_RP + 'apnea/'
IMG_SUFFIX = '.png'
# ========================================================================================


# ==================================== config for RQA ====================================
MY_LAMBDA = 15
FOLDER_SAVE_RQA = res + 'rqa/'
# =======================================================================================
