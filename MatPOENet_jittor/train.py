##########################################################################################
# Machine Environment Config

USE_CUDA = True

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging

from utils_jittor.utils import create_logger, copy_all_src
from ATSPTrainer import ATSPTrainer as Trainer
import warnings
warnings.filterwarnings('ignore')

##########################################################################################
# parameters

scale = 50

env_params = {
    'max_scale': scale + 1,
    'min_scale': scale,
    'problem_pool': ["atsp_triangle", "tsp_euc", "hcp", "3sat"],
    "init_solver": 'nn',
    "pos_embedding_dim": 512,

    'node_cnt': scale,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
}

model_params = {
    'pos_embedding_dim': env_params["pos_embedding_dim"],
    'embedding_dim': 512,
    'sqrt_embedding_dim': 512**(1/2),
    'encoder_layer_num': 8,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'argmax'
}

optimizer_params = {
    'optimizer': {
        'lr': 3*1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [2001, 2101],  # if further training is needed
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'epochs': 5,
    'train_episodes': 10*100,
    'train_batch_size': 150,
    'val_interval': 1,
    'val_dir': '../data/val_set/{}_2000',
    'logging': {
        'model_save_interval': 5,
        'img_save_interval': 5,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },
    'model_load': {
        'enable': False,   # enable loading pre-trained model
        # 'path': # directory path of pre-trained model and log files saved.
        # 'epoch': # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': f'train_mix{scale}',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()

def _print_config():
    logger = logging.getLogger('root')
    logger.info('USE_CUDA: {}'.format(USE_CUDA))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
