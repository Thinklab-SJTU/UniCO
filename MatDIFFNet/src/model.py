import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Any, Union, Tuple
from ml4co_kit import BaseModel, ATSPSolver, to_numpy
from .env import MatDIFFNetEnv
from .gnn_encoder import GNNEncoder
from .decoder import MatDIFFNetDecoder
from .diffusion_utils import CategoricalDiffusion, InferenceSchedule


def get_coords_by_dists(dists: Tensor) -> Union[Tensor, Tensor]:
    batch_size, nodes_num, _ = dists.shape
    coords = torch.arange(2*nodes_num) + 1 / (2*nodes_num)
    coords = coords.reshape(1, nodes_num, 2)
    coords = coords.expand(batch_size, nodes_num, 2)
    coords_source = coords[:, :, 0]
    coords_target = coords[:, :, 1]
    return coords_source, coords_target


class MatDIFFNetModel(BaseModel):
    def __init__(
        self,
        env: MatDIFFNetEnv,
        encoder: GNNEncoder,
        decoder: MatDIFFNetDecoder,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        # diffusion
        diffusion_schedule: str = "linear",
        inference_schedule: str = "cosine",
        diffusion_steps: int = 1000,
        inference_diffusion_steps: int = 50,
        parallel_sampling: int = 1,
        sequential_sampling: int = 1,
    ):      
        # super
        super(MatDIFFNetModel, self).__init__(
            env=env,
            model=encoder,
            lr_scheduler=lr_scheduler,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.env: MatDIFFNetEnv
        self.model: GNNEncoder
        self.decoder = decoder

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
        self, target_t: int, t: int, x0_pred_prob: Tensor, xt: Tensor
    ) -> Tensor:
        diffusion = self.diffusion

        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)

        if target_t > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
            Q_t = torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
        else:
            Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = (
            torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        )
        Q_bar_t_target = (
            torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)
        )

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)

        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
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
            xt = torch.bernoulli(sum_x_t_target_prob.clamp(0, 1))
        else:
            xt = sum_x_t_target_prob.clamp(min=0)

        return xt

    def guided_categorical_posterior(
        self,
        target_t: int,
        t: int,
        x0_pred_prob: Tensor,
        xt: Tensor,
        grad=None,
    ) -> Tensor:
        # xt: b, n, n
        if grad is None:
            grad = xt.grad
        with torch.no_grad():
            diffusion = self.diffusion
        if target_t is None:
            target_t = t - 1
        else:
            target_t = torch.from_numpy(target_t).view(1)

        if target_t > 0:
            Q_t = np.linalg.inv(diffusion.Q_bar[target_t]) @ diffusion.Q_bar[t]
            Q_t = (
                torch.from_numpy(Q_t).float().to(x0_pred_prob.device)
            )  # [2, 2], transition matrix
        else:
            Q_t = torch.eye(2).float().to(x0_pred_prob.device)
        Q_bar_t_source = (
            torch.from_numpy(diffusion.Q_bar[t]).float().to(x0_pred_prob.device)
        )
        Q_bar_t_target = (
            torch.from_numpy(diffusion.Q_bar[target_t]).float().to(x0_pred_prob.device)
        )

        xt_grad_zero, xt_grad_one = torch.zeros(xt.shape, device=xt.device).unsqueeze(
            -1
        ).repeat(1, 1, 1, 2), torch.zeros(xt.shape, device=xt.device).unsqueeze(
            -1
        ).repeat(
            1, 1, 1, 2
        )
        xt_grad_zero[..., 0] = (1 - xt) * grad
        xt_grad_zero[..., 1] = -xt_grad_zero[..., 0]
        xt_grad_one[..., 1] = xt * grad
        xt_grad_one[..., 0] = -xt_grad_one[..., 1]
        xt_grad = xt_grad_zero + xt_grad_one

        xt = F.one_hot(xt.long(), num_classes=2).float()
        xt = xt.reshape(x0_pred_prob.shape)  # [b, n, n, 2]

        # q(xt−1|xt,x0=0)pθ(x0=0|xt)
        x_t_target_prob_part_1 = torch.matmul(xt, Q_t.permute((1, 0)).contiguous())
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

        p_theta = torch.cat(
            (1 - sum_x_t_target_prob.unsqueeze(-1), sum_x_t_target_prob.unsqueeze(-1)),
            dim=-1,
        )
        p_phi = torch.exp(-xt_grad)
        posterior = (p_theta * p_phi) / torch.sum(
            (p_theta * p_phi), dim=-1, keepdim=True
        )

        if target_t > 0:
            xt = torch.bernoulli(posterior[..., 1].clamp(0, 1))
        else:
            xt = posterior[..., 1].clamp(min=0)
            
        return xt

    def categorical_denoise_step(
        self,
        dists: Tensor,
        xt: Tensor,
        t: Tensor,
        device: str,
        edge_index: Tensor = None,
        target_t: Tensor = None,
    ):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            xt: Tensor
            if xt.ndim == 2:
                xt = xt.unsqueeze(dim=0)
            xt = xt.to(self.device)

            xt_scale = (xt * 2 - 1).float()
            xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt_scale))

            # get nodes
            x1, x2 = get_coords_by_dists(dists)
            x1 = x1.float().to(self.device)
            x2 = x2.float().to(self.device)
        
            x0_pred = self.model.forward(
                x1=x1,
                x2=x2,
                dists=dists.float().to(device),
                graph=xt_scale.float().to(device),
                edge_index=edge_index.long().to(device) if edge_index is not None else None,
                timesteps=t.float().to(device),
            )

            x0_pred_prob = (
                x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            )
            
            x0_pred_prob = x0_pred_prob.to(self.device)
            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt

    @torch.enable_grad() 
    @torch.inference_mode(False)
    def guided_categorical_denoise_step(
        self, 
        dists: Tensor, 
        xt: Tensor, 
        t: Tensor, 
        device: str, 
        edge_index: Tensor=None, 
        target_t: Tensor=None
    ):
        xt = xt.float()  # b, n, n
        xt.requires_grad = True
        t = torch.from_numpy(t).view(1)
        if edge_index is not None: edge_index = edge_index.clone()

        # [b, 2, n, n]
        # with torch.inference_mode(False):
        ###############################################
        # scale to [-1, 1]
        xt_scale = (xt * 2 - 1)
        xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt_scale))
        # xt_scale = xt
        ###############################################

        # print(dists.shape, xt.shape)
        x0_pred = self.forward(
            dists.float().to(device),
            xt_scale.to(device),
            edge_index.long().to(device) if edge_index is not None else None,
            t.float().to(device),
        )

        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
        cost_est = (dists * x0_pred_prob[..., 1]).sum()
        cost_est.requires_grad_(True)
        cost_est.backward()
        
        assert xt.grad is not None

        xt.grad = nn.functional.normalize(xt.grad, p=2, dim=-1)
        xt = self.guided_categorical_posterior(target_t, t, x0_pred_prob, xt)

        return xt.detach()
      
    def solve_encode(self, dists: Tensor) -> Tensor:        
        device = dists.device
        heatmap_list = list()

        # dists & edge_index
        dists = dists.repeat(self.parallel_sampling, 1, 1)

        # heatmap
        batch_size = dists.shape[0]
        for _ in range(self.sequential_sampling):
            # diffusion xt
            xt = torch.randn(batch_size, self.env.nodes_num, self.env.nodes_num)
            if self.parallel_sampling > 1:
                xt = xt.repeat(self.parallel_sampling, 1, 1)
                xt = torch.randn_like(xt)
            xt = (xt > 0).long().to(device)

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
                    dists, xt, t1, device, edge_index=None, target_t=t2
                )

            heatmap_list.append(xt)
            
        heatmap = torch.cat(heatmap_list, dim=0)
        heatmap = heatmap.reshape(-1, self.env.nodes_num, self.env.nodes_num)
        return heatmap

    def train_test_encode(
        self, dists: Tensor, adj_matrix: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        # xt
        adj_matrix_onehot: Tensor = F.one_hot(adj_matrix.long(), num_classes=2)
        adj_matrix_onehot = adj_matrix_onehot.float()
        t = np.random.randint(1, self.diffusion.T + 1, dists.shape[0]).astype(int)
        xt = self.diffusion.sample(adj_matrix_onehot, t)    
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        # t
        t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

        # get nodes
        x1, x2 = get_coords_by_dists(dists)
        x1 = x1.float().to(self.device)
        x2 = x2.float().to(self.device)
        
        # x0_pred
        x0_pred = self.model.forward(
            x1=x1,
            x2=x2,
            dists=dists.float().to(self.device),
            graph=xt.float().to(self.device),
            edge_index=None,
            timesteps=t.float().to(self.device)
        )

        # loss
        loss = nn.CrossEntropyLoss()(x0_pred, adj_matrix.long())

        if self.env.mode == "train":
            return None, loss
        else:
            heatmap = self.solve_encode(dists)
            return heatmap, loss

    def encode(
        self, dists: Tensor, adj_matrix: Tensor = None
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if self.env.mode == "solve":
            return self.solve_encode(dists)
        else:
            return self.train_test_encode(dists, adj_matrix)
            
    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        # env mode
        self.env.mode = phase

        if phase == "train":
            # batch data is fake, just get batch length
            batch_size = len(batch)
            del batch
        
            # use env to get data
            dists, ref_tours = self.env.generate_data(batch_size)
            
            # process data
            adj_matrix = self.env.process_data(dists, ref_tours)
            
            # real share encoding step & calculate loss
            heatmap, loss = self.encode(dists=dists, adj_matrix=adj_matrix)
            metrics = {f"{phase}/loss": loss}
        
        else:
            # get val data from env
            val_data_dict = self.env.get_val_data()
            
            all_loss = 0
            metrics = dict()
            # validation for TSP, ATSP, SAT, HCP
            for k, v in val_data_dict.items():
                # get data   
                dists, ref_tours = v
                dists = torch.from_numpy(dists).to(self.device)
                ref_tours = torch.from_numpy(ref_tours).to(self.device)

                # process data
                self.env.nodes_num = dists.shape[-1]
                self.decoder.nodes_num = self.env.nodes_num
                adj_matrix = self.env.process_data(dists, ref_tours)

                # real share encoding step & calculate loss
                heatmap, loss = self.encode(dists=dists, adj_matrix=adj_matrix)
                
                all_loss += loss.item()
                
                # decoding
                solved_tours = self.decoder.decode(heatmap=heatmap, dists=dists)

                # calculate gap
                tmp_solver = ATSPSolver()
                tmp_solver.from_data(
                    dists=to_numpy(dists), tours=to_numpy(ref_tours), ref=True
                )
                solved_tours = solved_tours.reshape(-1, self.env.nodes_num + 1)
                tmp_solver.from_data(tours=solved_tours, ref=False)

                # Gap or Length
                if k == "TSP":
                    _, _, gap, _ = tmp_solver.evaluate(calculate_gap=True)
                    metrics.update({f"{phase}/tsp_gap": gap})
                if k == "ATSP":
                    _, _, gap, _ = tmp_solver.evaluate(calculate_gap=True)
                    metrics.update({f"{phase}/atsp_gap": gap})
                if k == "SAT":
                    costs_avg = tmp_solver.evaluate()
                    metrics.update({f"{phase}/sat_costs": costs_avg})
                if k == "HCP":
                    costs_avg = tmp_solver.evaluate()
                    metrics.update({f"{phase}/hcp_costs": costs_avg})
                if k == "VC":
                    costs_avg = tmp_solver.evaluate()
                    metrics.update({f"{phase}/vc_costs": costs_avg})
                
            metrics.update({f"{phase}/loss": all_loss})

        # log
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)   
        
        return loss if phase == "train" else metrics