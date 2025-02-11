
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
from logging import getLogger

from ATSPEnv import ATSPEnv as Env
from ATSPModel import ATSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
from utils.ATSProblemDef import load_single_problem_from_file
from tqdm import tqdm

class ATSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        self.best_score = 1e10
        self.best_cls_scores = None

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        val_interval = self.trainer_params['val_interval']
        n_nodes = self.env_params['min_scale']
        val_dir = self.trainer_params['val_dir'].format(n_nodes)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # Validation TODO
            # if epoch % val_interval == 0:
            #     cls = ['atsp', 'tsp2d', 'hcp', '3sat']
            #     val_score, val_score_aug, cls_scores = self._validation(dir=val_dir)
            #     self.result_log.append('val_score', epoch, val_score)
            #     self.result_log.append('val_score_aug', epoch, val_score_aug)
            #     self.result_log.append('val_score_atsp', epoch, cls_scores[0].item())
            #     self.result_log.append('val_score_euc', epoch, cls_scores[1].item())
            #     self.result_log.append('val_score_hcp', epoch, cls_scores[2].item())
            #     self.result_log.append('val_score_3sat', epoch, cls_scores[3].item())
            #     self.logger.info(f'val_score: {val_score:.4f}, val_score_aug: {val_score_aug:.4f}')
            #     for i in range(4):
            #         self.logger.info(f'{cls[i]}: {cls_scores[i]:.4f}')

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])
                # if epoch % val_interval == 0:
                #     util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                #                         self.result_log, labels=['val_score'])
                #     for cls in ['atsp', 'euc', 'hcp', '3sat']:
                #         util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                #                             self.result_log, labels=[f'val_score_{cls}'])

            if epoch % model_save_interval == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict,
                    f'{self.result_folder}/checkpoint-{epoch}.pt')

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _validation(self, n_instances=2000, dir='../data/val_set/20_2000'):
        no_aug_scores, aug_scores = 0, 0
        cls_scores = torch.zeros((4,)) # atsp, hcp, euc, 3sat
        problems= []
        # load all validation instances
        for i in range(n_instances):
            filename = os.path.join(dir, f'{i}.atsp')
            problem = load_single_problem_from_file(filename)
            problems.append(problem.cpu().numpy().tolist())

        batch_size = 100
        aug_factor = 8
        # batch_size = aug_factor * batch_size
        num_batches = n_instances // batch_size # 20
        for batch_idx in tqdm(range(num_batches)):
            zero_cnt = {'no_aug':0, 'aug':0} # for decisive problems
            batched_problems = problems[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            # print(batched_problems)
            # types = ['atsp', 'euc', 'hcp', '3sat']
            batched_problems = torch.tensor(batched_problems)
            batched_problems = batched_problems.repeat(aug_factor, 1, 1)
            self.model.eval()
            with torch.no_grad():
                self.env.load_problems_manual(batched_problems)
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward_pe(reset_state, self.env.pos_emb)

                # POMO Rollout
                ###############################################
                state, reward, done = self.env.pre_step()
                while not done:
                    selected, _ = self.model(state)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected)

                # Return
                ###############################################
                aug_reward = reward.reshape(aug_factor, batch_size, self.env.node_cnt)
                # shape: (augmentation, batch, pomo)

                max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
                # shape: (augmentation, batch)
                no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

                max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
                # shape: (batch,)
                aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

                aug_zero = (-max_aug_pomo_reward.float() < 1e-3).sum()
                no_aug_zero = (-max_pomo_reward[0, :].float() < 1e-3).sum()
                zero_cnt['aug'] += int(aug_zero)
                zero_cnt['no_aug'] += int(no_aug_zero)

                no_aug_scores += no_aug_score.item()
                aug_scores += aug_score.item()
                cls_scores[batch_idx // 5] += no_aug_score
            
        return no_aug_scores / num_batches, aug_scores / num_batches, cls_scores / (num_batches / 4)

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()

        self.env.load_problems_from_pool(batch_size)
        reset_state, _, _ = self.env.reset()

        self.model.pre_forward_pe(reset_state, self.env.pos_emb)

        prob_list = torch.zeros(size=(batch_size, self.env.node_cnt, 0))
        # shape: (batch, pomo, 0~)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()
    
    def _rollout_k_steps(self, k):
        state, reward, done = self.env.pre_step()
        for _ in range(k):
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)

            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        return prob_list, state, reward, done