import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from typing import List, Optional
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.data.dataset.dataset import Dataset, Interaction
from recbole.config.configurator import Config
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType
import torch.nn.functional as F


class LightGCL(GeneralRecommender):
    r"""LightGCL is a GCN-based recommender model.

    LightGCL guides graph augmentation by singular value decomposition (SVD) to not only
    distill the useful information of user-item interactions but also inject the global
    collaborative context into the representation alignment of contrastive learning.

    We implement the model following the original author with a pairwise training mode.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config: Config, dataset: Dataset):
        super(LightGCL, self).__init__(config, dataset)
        self._user: torch.Tensor = dataset.inter_feat[dataset.uid_field]
        self._item: torch.Tensor = dataset.inter_feat[dataset.iid_field]

        # load parameters info
        self.embed_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.dropout = config["dropout"]
        self.temp = config["temp"]
        self.lambda_1 = config["lambda1"]
        self.lambda_2 = config["lambda2"]
        self.q = config["q"]

        # get the normalized adjust matrix
        self.adj_norm = self.coo2tensor(self.create_adjust_matrix())

        # perform svd reconstruction
        svd_u, s, svd_v = torch.svd_lowrank(self.adj_norm, q=self.q)
        self.u_mul_s = svd_u @ (torch.diag(s))
        self.v_mul_s = svd_v @ (torch.diag(s))
        del s
        self.ut = svd_u.T
        self.vt = svd_v.T

        self.E_u_0 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((self.n_users, self.embed_dim), device=self.device)
            )
        )
        self.E_i_0 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty((self.n_items, self.embed_dim), device=self.device)
            )
        )

        self.E_u: Optional[torch.Tensor] = None
        self.E_i: Optional[torch.Tensor] = None
        self.G_u: Optional[torch.Tensor] = None
        self.G_i: Optional[torch.Tensor] = None
        
        self.restore_user_e: Optional[torch.Tensor] = None
        self.restore_item_e: Optional[torch.Tensor] = None

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def create_adjust_matrix(self) -> sp.coo_matrix:
        r"""Get the normalized interaction matrix of users and items.

        Returns:
            coo_matrix of the normalized interaction matrix.
        """
        ratings = np.ones(self._user.shape, dtype=np.float32)
        matrix = sp.csr_matrix(
            (ratings, (self._user.detach().numpy(), self._item.detach().numpy())),
            shape=(self.n_users, self.n_items),
        ).tocoo()
        rowD = np.squeeze(np.array(matrix.sum(1)), axis=1)
        colD = np.squeeze(np.array(matrix.sum(0)), axis=0)
        matrix.data /= np.sqrt(rowD[matrix.row] * colD[matrix.col])
        return matrix

    def coo2tensor(self, matrix: sp.coo_matrix) -> torch.Tensor:
        r"""Convert coo_matrix to tensor.

        Args:
            matrix (scipy.coo_matrix): Sparse matrix to be converted.

        Returns:
            torch.Tensor: Transformed sparse matrix.
        """
        indices = torch.vstack(
            (torch.from_numpy(matrix.row), torch.from_numpy(matrix.col))
        ).long()
        values = matrix.data.astype(np.float32)
        shape = matrix.shape
        x: torch.Tensor = torch.sparse_coo_tensor(
            indices, values, shape, device=self.device
        )
        x = x.coalesce()
        return x

    def sparse_dropout(
        self, sparse_tensor: torch.Tensor, dropout: float = 0.2
    ) -> torch.Tensor:
        assert sparse_tensor.is_sparse
        if dropout == 0.0:
            return sparse_tensor
        indices = sparse_tensor.indices()
        values = F.dropout(sparse_tensor.values(), p=dropout)
        size = sparse_tensor.size()
        new_sparse_tensor = torch.sparse_coo_tensor(
            indices, values, size, device=self.device
        )
        return new_sparse_tensor

    def forward(self):
        G_u_sum = self.E_u_0
        G_i_sum = self.E_i_0
        E_u_sum = self.E_u_0
        E_i_sum = self.E_i_0
        E_u_last = self.E_u_0
        E_i_last = self.E_i_0
        for layer in range(1, self.n_layers + 1):
            # GNN propagation
            vt_ei = self.vt @ E_i_last
            G_u_sum = G_u_sum + self.u_mul_s @ vt_ei
            ut_eu = self.ut @ E_u_last
            G_i_sum = G_i_sum + self.v_mul_s @ ut_eu
            
            E_u_last = torch.spmm(
                self.sparse_dropout(self.adj_norm, self.dropout),
                E_i_last,
            )
            E_u_sum = E_u_sum + E_u_last
            
            E_i_last = torch.spmm(
                self.sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1),
                E_u_last,
            )
            E_i_sum = E_i_sum + E_i_last
            
        # aggregate across layer
        self.E_u = E_u_sum
        self.E_i = E_i_sum
        self.G_u = G_u_sum
        self.G_i = G_i_sum

        return self.E_u, self.E_i

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user_list = interaction[self.USER_ID]
        pos_item_list = interaction[self.ITEM_ID]
        neg_item_list = interaction[self.NEG_ITEM_ID]
        self.forward()
        bpr_loss = self.calc_bpr_loss(
            user_list, pos_item_list, neg_item_list
        )
        ssl_loss = self.calc_ssl_loss(user_list, pos_item_list)
        total_loss = bpr_loss + ssl_loss
        return total_loss

    def calc_bpr_loss(
        self,
        user_list: torch.Tensor,
        pos_item_list: torch.Tensor,
        neg_item_list: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculate the pairwise Bayesian Personalized Ranking (BPR) loss and parameter regularization loss.

        Args:
            E_u_norm (torch.Tensor): Ego embedding of all users after forwarding.
            E_i_norm (torch.Tensor): Ego embedding of all items after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.
            neg_item_list (torch.Tensor): List of negative examples.

        Returns:
            torch.Tensor: Loss of BPR tasks and parameter regularization.
        """
        u_e = self.E_u[user_list]
        pi_e = self.E_i[pos_item_list]
        ni_e = self.E_i[neg_item_list]
        pos_scores = torch.mul(u_e, pi_e).sum(dim=1)
        neg_scores = torch.mul(u_e, ni_e).sum(dim=1)
        loss1 = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

        # reg loss
        loss_reg = 0
        for param in self.parameters():
            loss_reg += torch.norm(param, p=2)
        loss_reg *= self.lambda_2
        return loss1 + loss_reg

    def calc_ssl_loss(
        self,
        user_list: torch.Tensor,
        pos_item_list: torch.Tensor,
    ) -> torch.Tensor:
        r"""Calculate the loss of self-supervised tasks.

        Args:
            E_u_norm (torch.Tensor): Ego embedding of all users in the original graph after forwarding.
            E_i_norm (torch.Tensor): Ego embedding of all items in the original graph after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """
        neg_score = (
            torch.logsumexp(self.G_u[user_list] @ self.E_u[user_list].T / self.temp, dim=1).mean()
            + torch.logsumexp(self.G_i[pos_item_list] @ self.E_i[pos_item_list].T / self.temp, dim=1).mean()
        )
        pos_score = (
            torch.clamp(
                (self.G_u[user_list] * self.E_u[user_list]).sum(1) / self.temp,
                -5.0,
                5.0,
            )
        ).mean() + (
            torch.clamp(
                (self.G_i[pos_item_list] * self.E_i[pos_item_list]).sum(1) / self.temp,
                -5.0,
                5.0,
            )
        ).mean()
        ssl_loss = -pos_score + neg_score
        return self.lambda_1 * ssl_loss

    def predict(self, interaction: Interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        user = self.restore_user_e[interaction[self.USER_ID]]
        item = self.restore_item_e[interaction[self.ITEM_ID]]
        return torch.sum(user * item, dim=1)

    def full_sort_predict(self, interaction: Interaction):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        user = self.restore_user_e[interaction[self.USER_ID]]
        return user.matmul(self.restore_item_e.T)
