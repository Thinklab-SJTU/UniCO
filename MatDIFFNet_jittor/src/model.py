import os
import jittor as jt
import numpy as np
from jittor import nn, Var
from typing import Any, Union, Tuple
from ml4co_kit import ATSPSolver
from tqdm import trange
from .env import MatDIFFNetEnv
from .gnn_encoder import GNNEncoder
from .decoder import MatDIFFNetDecoder
from .diffusion_utils import CategoricalDiffusion, InferenceSchedule


def get_coords_by_dists(dists: Var) -> Tuple[Var, Var]:
    batch_size, nodes_num, _ = dists.shape
    coords = jt.arange(2*nodes_num) + 1 / (2*nodes_num)
    coords = coords.reshape(1, nodes_num, 2)
    coords = coords.expand(batch_size, nodes_num, 2)
    coords_source = coords[:, :, 0]
    coords_target = coords[:, :, 1]
    return coords_source, coords_target

class MatDIFFNetModel(object):
    def __init__(
        self,
        env: MatDIFFNetEnv,
        encoder: GNNEncoder,
        decoder: MatDIFFNetDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        val_evry_n_epochs: int = 1,
        ckpt_save_dir: str = "your/save/dir",
        # diffusion
        diffusion_schedule: str = "linear",
        inference_schedule: str = "cosine",
        diffusion_steps: int = 1000,
        inference_diffusion_steps: int = 50,
        parallel_sampling: int = 1,
        sequential_sampling: int = 1,
        # cuda
        use_cuda: bool = True,
        # pretrained weight
        pretrained_path: str = None
    ):      
        self.env: MatDIFFNetEnv = env
        self.model: GNNEncoder = encoder
        self.decoder = decoder

        self.max_epochs = max_epochs
        self.val_evry_n_epochs = val_evry_n_epochs
        self.best_avg_obj_val = 1e6
        os.makedirs(ckpt_save_dir, exist_ok=True)
        self.ckpt_save_path = os.path.join(ckpt_save_dir, "checkpoint-{}-{}.pkl")

        jt.flags.use_cuda = use_cuda # set device for jittor globally

        if lr_scheduler == "constant":
            self.optimizer = jt.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif lr_scheduler == "cosine-decay":
            self.optimizer = jt.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            self.lr_scheduler = jt.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.env.train_data_size // self.env.train_batch_size * self.max_epochs,
                eta_min=0.0,
            )
        
        # load pretrained weight
        if pretrained_path is not None:
            self.model.load(pretrained_path)
            print(f"Pretrained weights loaded: {pretrained_path}.")

        # diffusion
        self.diffusion_schedule = diffusion_schedule
        self.inference_schedule = inference_schedule
        self.diffusion_steps = diffusion_steps
        self.inference_diffusion_steps = inference_diffusion_steps
        self.diffusion = CategoricalDiffusion(
            T=self.diffusion_steps, schedule=self.diffusion_schedule
        )
        self.parallel_sampling = parallel_sampling
        self.sequential_sampling = sequential_sampling
            
        # record solved tours
        self.solved_tours_list = list()
    
    def categorical_posterior(
        self, target_t: int, t: int, x0_pred_prob: Var, xt: Var
    ) -> Var:
        diffusion = self.diffusion

        if target_t is None:
            target_t = t - 1
        else:
            target_t = jt.Var(target_t).view(1)

        if target_t > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
            Q_t = jt.float32(Q_t)
        else:
            Q_t = jt.init.eye(2, jt.float32)
        Q_bar_t_source = (
            jt.float32(diffusion.Q_bar[t])
        )
        Q_bar_t_target = (
            jt.float32(diffusion.Q_bar[target_t])
        )

        xt = nn.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)

        x_t_target_prob_part_1 = jt.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (
            x_t_target_prob_part_1 * x_t_target_prob_part_2
        ) / x_t_target_prob_part_3

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (
            x_t_target_prob_part_1 * x_t_target_prob_part_2_new
        ) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        if target_t > 0:
            xt = jt.bernoulli(jt.clamp(sum_x_t_target_prob, 0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min_v=0)

        return xt

    def guided_categorical_posterior(
        self,
        target_t: int,
        t: int,
        x0_pred_prob: Var,
        xt: Var,
        grad=None,
    ) -> Var:
        # xt: b, n, n
        if grad is None:
            grad = xt.grad
        with jt.no_grad():
            diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = jt.Var(target_t).view(1)

        if target_t > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
            Q_t = (
                jt.float32(Q_t)
            )  # [2, 2], transition matrix
        else:
            Q_t = jt.init.eye(2, "float32")
        Q_bar_t_source = (
            jt.float32(diffusion.Q_bar[t])
        )
        Q_bar_t_target = (
            jt.float32(diffusion.Q_bar[target_t])
        )

        xt_grad_zero, xt_grad_one = jt.zeros(xt.shape).unsqueeze(-1).repeat(1, 1, 1, 2), \
                                    jt.zeros(xt.shape).unsqueeze(-1).repeat(1, 1, 1, 2)
        xt_grad_zero[..., 0] = (1 - xt) * grad
        xt_grad_zero[..., 1] = -xt_grad_zero[..., 0]
        xt_grad_one[..., 1] = xt * grad
        xt_grad_one[..., 0] = -xt_grad_one[..., 1]
        xt_grad = xt_grad_zero + xt_grad_one

        xt = nn.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)  # [b, n, n, 2]

        # q(xt−1|xt,x0=0)pθ(x0=0|xt)
        x_t_target_prob_part_1 = jt.matmul(xt, Q_t.permute((1, 0)).contiguous())
        x_t_target_prob_part_2 = Q_bar_t_target[0]
        x_t_target_prob_part_3 = (Q_bar_t_source[0] * xt).sum(dim=-1, keepdim=True)

        x_t_target_prob = (
            x_t_target_prob_part_1 * x_t_target_prob_part_2
        ) / x_t_target_prob_part_3  # [b, n, n, 2]

        sum_x_t_target_prob = x_t_target_prob[..., 1] * x0_pred_prob[..., 0]

        # q(xt−1|xt,x0=1)pθ(x0=1|xt)
        x_t_target_prob_part_2_new = Q_bar_t_target[1]
        x_t_target_prob_part_3_new = (Q_bar_t_source[1] * xt).sum(dim=-1, keepdim=True)

        x_t_source_prob_new = (
            x_t_target_prob_part_1 * x_t_target_prob_part_2_new
        ) / x_t_target_prob_part_3_new

        sum_x_t_target_prob += x_t_source_prob_new[..., 1] * x0_pred_prob[..., 1]

        p_theta = jt.cat(
            (1 - sum_x_t_target_prob.unsqueeze(-1), sum_x_t_target_prob.unsqueeze(-1)),
            dim=-1,
        )
        p_phi = jt.exp(-xt_grad)
        posterior = (p_theta * p_phi) / jt.sum(
            (p_theta * p_phi), dim=-1, keepdim=True
        )

        if target_t > 0:
            xt = jt.bernoulli(jt.clamp(posterior[..., 1], 0, 1))
        else:
            xt = jt.clamp(posterior[..., 1], min_v=0)
            
        return xt

    def categorical_denoise_step(
        self,
        dists: Var,
        xt: Var,
        t: Var,
        edge_index: Var = None,
        target_t: Var = None,
    ):
        with jt.no_grad():
            t = jt.Var(t).view(1)
            xt: Var
            if xt.ndim == 2:
                xt = xt.unsqueeze(dim=0)

            xt_scale = (xt * 2 - 1).float()
            xt_scale = xt_scale * (1.0 + 0.05 * jt.rand_like(xt_scale))

            # get nodes
            x1, x2 = get_coords_by_dists(dists)
            x1 = x1.float()
            x2 = x2.float()
            
            x0_pred = self.model.execute(
                x1=x1,
                x2=x2,
                dists=dists.float(),
                graph=xt_scale.float(),
                edge_index=edge_index.long() if edge_index is not None else None,
                timesteps=t.float(),
            )

            x0_pred_prob = (
                x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            )
            
            x0_pred_prob = x0_pred_prob
            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt
  
    def solve_encode(self, dists: Var) -> Var:        
        heatmap_list = list()

        # dists & edge_index
        dists = dists.repeat(self.parallel_sampling, 1, 1)

        # heatmap
        batch_size = dists.shape[0]
        for _ in range(self.sequential_sampling):
            # diffusion xt
            xt = jt.randn(batch_size, self.env.nodes_num, self.env.nodes_num)
            if self.parallel_sampling > 1:
                xt = xt.repeat(self.parallel_sampling, 1, 1)
                xt = jt.randn_like(xt)
            xt = (xt > 0).long()

            # time schedule
            time_schedule = InferenceSchedule(
                inference_schedule=self.inference_schedule,
                T=self.diffusion.T,
                inference_T=self.inference_diffusion_steps,
            )

            # Diffusion iterations
            for i in range(self.inference_diffusion_steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)
                # [B, N, N], heatmap score
                xt = self.categorical_denoise_step(
                    dists, xt, t1, edge_index=None, target_t=t2
                )

            heatmap_list.append(xt)
            
        heatmap = jt.concat(heatmap_list, dim=0)
        heatmap = heatmap.reshape(-1, self.env.nodes_num, self.env.nodes_num)
        return heatmap

    def train_test_encode(
        self, dists: Var, adj_matrix: Var = None,
    ) -> Tuple[Var, Var]:
        # xt
        adj_matrix_onehot: Var = nn.one_hot(adj_matrix.long(), num_classes=2)
        adj_matrix_onehot = adj_matrix_onehot.float()
        t = np.random.randint(1, self.diffusion.T + 1, dists.shape[0]).astype(int)
        xt = self.diffusion.sample(adj_matrix_onehot, t)    
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * jt.rand_like(xt))

        # t
        t = jt.float32(t).view(adj_matrix.shape[0])

        # get nodes
        x1, x2 = get_coords_by_dists(dists)
        x1 = x1.float()
        x2 = x2.float()
        
        # x0_pred
        x0_pred = self.model.execute(
            x1=x1,
            x2=x2,
            dists=dists.float(),
            graph=xt.float(),
            edge_index=None,
            timesteps=t.float()
        )

        # loss
        loss = nn.CrossEntropyLoss()(x0_pred, adj_matrix.long())

        if self.env.mode == "train":
            return None, loss
        else:
            heatmap = self.solve_encode(dists)
            return heatmap, loss

    def encode(
        self, dists: Var, adj_matrix: Var = None
    ) -> Union[Tuple[Var, Var], Var]:
        if self.env.mode == "solve":
            return self.solve_encode(dists)
        else:
            return self.train_test_encode(dists, adj_matrix)
            
    def model_train(self):
        num_batches_per_epoch = self.env.train_data_size // self.env.train_batch_size
        tr = trange(self.max_epochs)
        for epoch in tr:  
            for batch_idx in range(num_batches_per_epoch):
                # use env to get data
                dists, ref_tours = self.env.generate_data(self.env.train_batch_size)
                adj_matrix = self.env.process_data(dists, ref_tours)

                # generate heatmap & calculate loss to update model parameters
                heatmap, loss = self.encode(dists=dists, adj_matrix=adj_matrix)
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                self.optimizer.step()
                tr.set_description(f"Epoch {epoch} ({batch_idx}/{num_batches_per_epoch}), loss={loss:.4f}")

            # validation
            if epoch % self.val_evry_n_epochs == 0:
                print("Valiadtion...")
                avg_obj_val = self.model_eval()
                if avg_obj_val < self.best_avg_obj_val:
                    self.best_avg_obj_val = avg_obj_val
                    self.model.save(self.ckpt_save_path.format(epoch, f"{avg_obj_val:.4f}"))
                    print(f"Avg. obj: {avg_obj_val}, saved new best!")
                else:
                    print(f"Avg. obj: {avg_obj_val}")

    def model_eval(self):    
        # get val data from env
        val_data_dict = self.env.get_val_data()
        
        # all_loss = 0
        all_obj = 0
        # validation for TSP, ATSP, SAT, HCP
        for k, v in val_data_dict.items():
            # get data   
            dists, ref_tours = v
            dists = jt.Var(dists)
            ref_tours = jt.Var(ref_tours)

            # process data
            self.env.nodes_num = dists.shape[-1]
            self.decoder.nodes_num = self.env.nodes_num
            adj_matrix = self.env.process_data(dists, ref_tours)

            # real share encoding step & calculate loss
            heatmap = self.solve_encode(dists=dists)
                        
            # decoding
            solved_tours = self.decoder.decode(heatmap=heatmap, dists=dists)

            # calculate gap
            tmp_solver = ATSPSolver()
            tmp_solver.from_data(
                dists=dists.numpy(), tours=ref_tours.numpy(), ref=True
            )
            solved_tours = solved_tours.reshape(-1, self.env.nodes_num + 1)
            tmp_solver.from_data(tours=solved_tours, ref=False)

            # Gap or Length
            if k == "TSP":
                costs_avg, _, gap, _ = tmp_solver.evaluate(calculate_gap=True)
                # metrics.update({f"tsp_obj": costs_avg})
                all_obj += costs_avg
                print(f"[TSP] obj: {costs_avg:.4f}, gap: {gap:.4f}%")
            if k == "ATSP":
                costs_avg, _, gap, _ = tmp_solver.evaluate(calculate_gap=True)
                all_obj += costs_avg
                # metrics.update({f"atsp_obj": costs_avg})
                print(f"[ATSP] obj: {costs_avg:.4f}, gap: {gap:.4f}%")
            if k == "SAT":
                costs_avg = tmp_solver.evaluate()
                all_obj += costs_avg
                # metrics.update({f"sat_obj": costs_avg})
                print(f"[SAT] obj: {costs_avg:.4f}")
            if k == "HCP":
                costs_avg = tmp_solver.evaluate()
                all_obj += costs_avg
                # metrics.update({f"hcp_obj": costs_avg})
                print(f"[HCP] obj: {costs_avg:.4f}")
            if k == "VC":
                costs_avg = tmp_solver.evaluate()
                all_obj += costs_avg
                # metrics.update({f"vc_obj": costs_avg})
                print(f"[VC] obj: {costs_avg:.4f}")
        # metrics.update({f"loss": all_loss})
        avg_obj = all_obj / len(val_data_dict)

        return avg_obj