# -------------root path----------------------------------- #
ROOT_PATH = "D:\\resume_project"


# ------------------------------------------------ #
RNN_DATA_PATH = ROOT_PATH + '\\resume_dataset'
RNN_DATA_DATA_PATH1 = RNN_DATA_PATH + '\\data1'
RNN_DATA_LABEL_PATH1 = RNN_DATA_PATH + '\\label1'
RNN_DATA_DATA_PATH2 = RNN_DATA_PATH + '\\data2'
RNN_DATA_LABEL_PATH2 = RNN_DATA_PATH + '\\label2'
RNN_DATA_DATA_PATH3 = RNN_DATA_PATH + '\\data3'
RNN_DATA_LABEL_PATH3 = RNN_DATA_PATH + '\\label3'

# -------------other resource----------------------------------- #
USER_DIC_PATH = ROOT_PATH + '\\material\\user_dic.txt'
TITLE_LIST_PATH = ROOT_PATH + '\\material\\title_list.txt'
STOP_LIST_PATH = ROOT_PATH + '\\material\\stop_list.txt'

# -------------word2vec path----------------------------------- #
word2vec_model_path_2021_2_5 = ROOT_PATH + '\\word2vec_update_2021_2_5'
# ---------------------model path--------------------------- #
muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec = ROOT_PATH + \
        '\\output_models\\text_cnn_model_word2vec_API_mode_update2_zhwiki_word2vec.h5'
FNN_MODEL_PATH = ROOT_PATH + '\\output_models\\fnn_11_30'
BBRNN_MODEL_PATH = ROOT_PATH + '\\b_brnn_model_11-29'
HYBRID_MODEL_DYNAMIC_WEIGHT_PATH = ROOT_PATH + '\\hybrid_model_dynamic_weight_12_1'

FNN_100_PATH = ROOT_PATH + '\\output_models\\fnn_100'
FNN_200_PATH = ROOT_PATH + '\\output_models\\fnn_200'
FNN_300_PATH = ROOT_PATH + '\\output_models\\fnn_300'
FNN_400_PATH = ROOT_PATH + '\\output_models\\fnn_400'
FNN_500_PATH = ROOT_PATH + '\\output_models\\fnn_500'
FNN_600_PATH = ROOT_PATH + '\\output_models\\fnn_600'
FNN_700_PATH = ROOT_PATH + '\\output_models\\fnn_700'
JOINT_100_PATH = ROOT_PATH + '\\output_models\\joint_100'
JOINT_200_PATH = ROOT_PATH + '\\output_models\\joint_200'
JOINT_300_PATH = ROOT_PATH + '\\output_models\\joint_300'
JOINT_400_PATH = ROOT_PATH + '\\output_models\\joint_400'
JOINT_500_PATH = ROOT_PATH + '\\output_models\\joint_500'
JOINT_600_PATH = ROOT_PATH + '\\output_models\\joint_600'
JOINT_700_PATH = ROOT_PATH + '\\output_models\\joint_700'
BRNN_100_PATH = ROOT_PATH + '\\output_models\\brnn_100'
BRNN_200_PATH = ROOT_PATH + '\\output_models\\brnn_200'
BRNN_300_PATH = ROOT_PATH + '\\output_models\\brnn_300'
BRNN_400_PATH = ROOT_PATH + '\\output_models\\brnn_400'
BRNN_500_PATH = ROOT_PATH + '\\output_models\\brnn_500'
BRNN_600_PATH = ROOT_PATH + '\\output_models\\brnn_600'
BRNN_700_PATH = ROOT_PATH + '\\output_models\\B-BRNN-IMPROVED-BY-FEATURE-INTEGRATION'
DW_HYBRID_100_PATH = ROOT_PATH + '\\output_models\\dynamic_weight_hybrid_model_100'
DW_HYBRID_200_PATH = ROOT_PATH + '\\output_models\\dynamic_weight_hybrid_model_200'
DW_HYBRID_300_PATH = ROOT_PATH + '\\output_models\\dynamic_weight_hybrid_model_300'
DW_HYBRID_400_PATH = ROOT_PATH + '\\output_models\\dynamic_weight_hybrid_model_400'
DW_HYBRID_500_PATH = ROOT_PATH + '\\output_models\\dynamic_weight_hybrid_model_500'
DW_HYBRID_600_PATH = ROOT_PATH + '\\output_models\\dynamic_weight_hybrid_model_600'
DW_HYBRID_700_PATH = ROOT_PATH + '\\output_models\\dynamic_weight_hybrid_model_700'
NOISE_10_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_10_brnn_model'
NOISE_9_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_9_brnn_model'
NOISE_8_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_8_brnn_model'
NOISE_7_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_7_brnn_model'
NOISE_6_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_6_brnn_model'
NOISE_5_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_5_brnn_model'
NOISE_4_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_4_brnn_model'
NOISE_3_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_3_brnn_model'
NOISE_2_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_2_brnn_model'
NOISE_1_BRNN_PATH = ROOT_PATH + '\\output_models\\noise_1_brnn_model'

# ---------------------------------------------------------searchPattern--------------------------------------------------#
DEL_SPECIAL_PAT = r'[^\u4E00-\u9FD5\d\w\]\[://@.]+'  # remove special character
DEL_SPECIAL_PAT2 = r'[^\u4E00-\u9FD5\d\w\]\[:：（）()_-——//@.]+'
CORRECT_NAME = r'[a-zA-Z]+'  # detect muti english character
DEL_SPECIAL_PAT1 = r'[^\u4E00-\u9FD5\d\w\]\[://@. \n]+'
CORRECT_LINK = r'([a-zA-Z]+://[^\s]*[.com|.cn|.edu|.org|.vip|.top|.net](/[a-zA-Z\d_]+)*)'
CORRECT_LINK1 = r'([wW]{3}.[^\s]*[.com|.cn|.edu|.org|.vip|.top|.net](/[a-zA-Z\d_]+)*)'
