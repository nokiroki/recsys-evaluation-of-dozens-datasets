"""SASRec model module."""
import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    """PointWise MLP module."""

    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        """Initialize the PointWise MLP module.

        Args:
            hidden_units (int): hidden size.
            dropout_rate (float): dropout rate.
        """
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            inputs (torch.Tensor): input data.

        Returns:
            torch.Tensor: output.
        """
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SASRec(torch.nn.Module):
    """SASRec torch model."""

    def __init__(
        self,
        user_num: int,
        item_num: int,
        device: torch.device,
        hidden_units: int,
        maxlen: int,
        dropout_rate: float,
        num_heads: int,
        num_blocks: int,
    ) -> None:
        """Init SASRec model.

        Args:
            user_num (int): number of users.
            item_num (int): number of items.
            device (torch.device): torch device to train.
            hidden_units (int): hidden size of model.
            maxlen (int): max length of sequence.
            dropout_rate (float): rate of dropout.
            num_heads (int): number of heads in transformer.
            num_blocks (int): number of transformer blocks.
        """
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = device

        # TODO: loss += l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, hidden_units, padding_idx=0
        )
        self.pos_emb = torch.nn.Embedding(maxlen, hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                hidden_units, num_heads, dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(hidden_units, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs: list[list[int]]) -> torch.Tensor:
        """Transform sequence to embeddings.

        Args:
            log_seqs (list[list[int]]): input data.

        Returns:
            torch.Tensor: transformed data.
        """
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i, _ in enumerate(self.attention_layers):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(
        self,
        user_ids: list[int],
        log_seqs: list[list[int]],
        pos_seqs: list[list[int]],
        neg_seqs: list[list[int]],
    ):  # for training
        """Forward method.

        Args:
            user_ids (list[int]): unused data.
            log_seqs (list[list[int]]): input data.
            pos_seqs (list[list[int]]): positive examples.
            neg_seqs (list[list[int]]): negative examples.

        Returns:
            (torch.Tensor, torch.Tensor): output data.
        """
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        """Predict items.

        Args:
            user_ids (list[int]): unused data.
            log_seqs (list[list[int]]): input data.
            item_indices (list[int]): item indices.

        Returns:
            list[float]: logits.
        """
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(
            torch.LongTensor(item_indices).to(self.dev)
        )  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
