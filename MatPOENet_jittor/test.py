##########################################################################################
# Machine Environment Config

USE_CUDA = True


##########################################################################################
# Path Config

import os
import sys
import jittor as jt
import numpy as np
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging

from utils_jittor.utils import create_logger, copy_all_src
from ATSPTester import ATSPTester as Tester


##########################################################################################
# parameters

WEIGHT_DICT = {
    50: "path/to/your/jittor/weight.pkl",
    100: "path/to/your/jittor/weight.pkl"
}

env_params = {
    'max_scale': None,
    'min_scale': None,
    "init_solver": 'nn', # choices [None, "rand", "nn", "ni", "fi"]:
    "pos_embedding_dim": 512,
    'node_cnt': 50,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
    'pomo_size': None  # same as node_cnt
}

model_params = {
    'pos_embedding_dim': env_params["pos_embedding_dim"],
    'embedding_dim': 512,
    'sqrt_embedding_dim': 512**(1/2),
    'encoder_layer_num': 8, #5 if env_params['node_cnt'] > 50 else 8,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'softmax'
}

tester_params = {
    'use_cuda': USE_CUDA,
    'model_load': WEIGHT_DICT[env_params['node_cnt']],
    # 'model_load': f'../ckpts/MatNet-POE_{env_params["node_cnt"]}.pt',
    'saved_problem_folder': f'/mnt/nas-new/home/panwenzheng/coformer/test_set/{env_params["node_cnt"]}_10000',
    'saved_problem_filename': '{}.atsp',
    'file_count': 100,
    'test_batch_size': 100,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 10,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': f'test_POE_{env_params["node_cnt"]}',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    
    jt.misc.set_global_seed(1234)

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
