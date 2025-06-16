import sys
sys.path.append("../")

from dataclasses import dataclass
import jittor as jt
import numpy as np
from utils_jittor.ATSProblemDef import get_random_problems
from utils_jittor.base_methods import *
from utils_jittor.generate_data import *
from utils_jittor.positional_encoding import *


@dataclass
class Reset_State:
    problems: jt.Var
    # shape: (batch, node, node)


@dataclass
class Step_State:
    BATCH_IDX: jt.Var
    POMO_IDX: jt.Var
    # shape: (batch, pomo)
    current_node: jt.Var = None
    # shape: (batch, pomo)
    ninf_mask: jt.Var = None
    # shape: (batch, pomo, node)


class ATSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.node_cnt = env_params['node_cnt']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # STEP-State
        ####################################
        self.step_state = None

        # multiple scales
        self.min_scale = env_params["min_scale"]
        self.max_scale = env_params["max_scale"]

        self.pos_emb_dim = self.env_params["pos_embedding_dim"]

        self.problem_type_idx = 0

    def load_problems_from_pool(self, batch_size):
        self.batch_size = batch_size
        n_problem = len(self.env_params["problem_pool"])
        self.problem_type_idx = np.random.randint(n_problem)
        problem_type = self.env_params["problem_pool"][self.problem_type_idx]
        if problem_type == "atsp_triangle":
            self.node_cnt = np.random.randint(
                self.env_params["min_scale"], 
                self.env_params["max_scale"])
            problem_gen_params = self.env_params['problem_gen_params']
            problems = get_random_problems(batch_size, self.node_cnt).cpu().numpy()
        elif problem_type == "tsp_euc":
            self.node_cnt = np.random.randint(
                self.env_params["min_scale"], 
                self.env_params["max_scale"])
            problems = [gen_Euclidean(self.node_cnt, 2) for _ in range(self.batch_size)]
        elif problem_type == "hcp":
            self.node_cnt = np.random.randint(
                self.env_params["min_scale"], 
                self.env_params["max_scale"])
            problems = [gen_hcp(self.node_cnt, np.random.rand() * 0.2 + 0.1) for _ in range(self.batch_size)]
        elif problem_type == "3sat":
            self.node_cnt = self.env_params["node_cnt"]
            if self.node_cnt == 20:
                num_clauses = [3, 2, 2]
                num_vars = [3, 4, 5]
            elif self.node_cnt == 50:
                num_clauses = [3, 4, 5, 6, 7]
                num_vars = [6, 5, 5, 4, 3]
            elif self.node_cnt == 100:
                num_clauses = [9, 8, 7, 6, 5]
                num_vars = [5, 6, 7, 8, 9]
            rand_idx = np.random.randint(0, len(num_clauses))
            self.node_cnt = 2 * num_clauses[rand_idx] * num_vars[rand_idx] + num_clauses[rand_idx]
            problems = [gen_3sat(num_clauses[rand_idx], num_vars[rand_idx]) for _ in range(self.batch_size)]
        else:
            raise NotImplementedError("Problem type {} is not implemented.".format(problem_type))
        self.BATCH_IDX = jt.arange(self.batch_size)[:, None].expand(self.batch_size, self.node_cnt)
        self.POMO_IDX = jt.arange(self.node_cnt)[None, :].expand(self.batch_size, self.node_cnt)
        # positional encoding
        self.pos_emb = make_positional_encoding_cosh_recur(self.pos_emb_dim, self.node_cnt, scaler=100)
        self.pos_emb = self.pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # initial tours
        solver = BaseSolver(self.node_cnt)
        if self.env_params["init_solver"] == None:
            pass
        elif self.env_params["init_solver"] == "nn":
            f_solver = solver.solve_nearest_neighbor 
        elif self.env_params["init_solver"] == "lkh":
            f_solver = solver.solve_lkh
        elif self.env_params["init_solver"] == "rand":
            f_solver = solver.solve_rand_perm
        elif self.env_params["init_solver"] == "fi":
            f_solver = solver.solve_farthest_insertion
        else:
            raise NotImplementedError("Solver {} is not implemented.".format(self.env_params["init_solver"]))
        if self.env_params["init_solver"]:
            for i in range(self.batch_size):
                f_solver(problems[i])
                init_tour = np.array(solver.get_path(), dtype=np.int64)
                problems[i] = problems[i][init_tour, :][:, init_tour]
        # problems
        self.problems = jt.Var(np.array(problems))

    def load_problems_manual(self, problems):
        # problems.shape: (batch, node, node)

        self.batch_size = problems.size(0)
        self.node_cnt = problems.size(1)
        self.BATCH_IDX = jt.arange(self.batch_size)[:, None].expand(self.batch_size, self.node_cnt)
        self.POMO_IDX = jt.arange(self.node_cnt)[None, :].expand(self.batch_size, self.node_cnt)

        self.pos_emb = make_positional_encoding_cosh_recur(self.pos_emb_dim, self.node_cnt, scaler=100)
        # self.pos_emb = make_positional_encoding_zero(self.pos_emb_dim, self.node_cnt)
        self.pos_emb = self.pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # initial tours
        solver = BaseSolver(self.node_cnt)
        if self.env_params["init_solver"] == None:
            pass
        elif self.env_params["init_solver"] == "nn":
            f_solver = solver.solve_nearest_neighbor
        elif self.env_params["init_solver"] == "lkh":
            f_solver = solver.solve_lkh
        elif self.env_params["init_solver"] == "fi":
            f_solver = solver.solve_farthest_insertion
        else:
            raise NotImplementedError("Solver {} is not implemented.".format(self.env_params["init_solver"]))
        if self.env_params["init_solver"]:
            for i in range(self.batch_size):
                f_solver(problems[i])
                init_tour = np.array(solver.get_path(), dtype=np.int64)
                problems[i] = problems[i][init_tour, :][:, init_tour]
        # problems
        self.problems = problems
        # shape: (batch, node, node)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = jt.empty((self.batch_size, self.node_cnt, 0), dtype=jt.int32)
        # shape: (batch, pomo, 0~)

        self._create_step_state()

        reward = None
        done = False
        return Reset_State(problems=self.problems), reward, done

    def _create_step_state(self):
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = jt.zeros((self.batch_size, self.node_cnt, self.node_cnt))
        # shape: (batch, pomo, node)

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, node_idx):
        # node_idx.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = node_idx
        # shape: (batch, pomo)
        self.selected_node_list = jt.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~node)

        self._update_step_state()
        
        # returning values
        done = (self.selected_count == self.node_cnt)
        if done:
            reward = -self._get_total_distance()  # Note the MINUS Sign ==> We MAXIMIZE reward
            # shape: (batch, pomo)
        else:    
            reward = None
        return self.step_state, reward, done

    def _update_step_state(self):
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

    def _get_total_distance(self):

        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.node_cnt, self.node_cnt)
        # shape: (batch, pomo, node)

        selected_cost = self.problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_distance
