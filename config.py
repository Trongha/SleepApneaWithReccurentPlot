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
RR_PER_RECURRENCE_PLOTS = 450
STEP_SIZE = 10
DIMENSION = 6
TAU = 10
PERCENT_DOT = 0.3

# ===> 1 Rp co 25 statePhase
FOLDER_SAVE_RP = res + 'rpImage/'
IMG_SUFFIX = '.png'
# ========================================================================================


# ==================================== config for RQA ====================================
EPSILON = 0.5
DISTANCE_NORM = 2
MY_LAMBDA = 15
FOLDER_SAVE_RQA = res + 'rqa/'
# =======================================================================================
