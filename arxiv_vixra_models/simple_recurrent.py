from copy import deepcopy
import json
from typing import Dict, Tuple, Sequence, Optional, Union
from typing_extensions import Literal
import warnings

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
import wandb


class LitRNNLoggingBaseAV(pl.LightningModule):
    """Custom PyTorch Lightning base class for recurrent architectures
    for arxiv/vixra classification with logging. Defining common methods
    for train, validation, and test steps.

    Description
    ----------
    Written such that one-hot and embedding models need only subclass
    LitRNNLoggingBaseAV with __init__ method overwritten.  Expects
    fully-connected layers after recurrent structure to be in a ModuleList
    instance assigned to an fc_layers attr, any Embedding layer to be
    similarly assigned to an embedding attr, and the recurrent layers
    to be assign to an rnn attr.

    Args
    ----------
    `seq_len`: int
        Length of sequences passed to the recurrent architecture.
    `lr`: float, default = 1e-3
        Learning rate for Adam optimizer.
    `l2_reg`: float, default = 0.
        Parameter controlling strength of l2 regularization on fully-connected
        layer weights.
    `hidden_strategy`: in ('last', 'max', 'mean', 'concat'), default = 'last'
        Determines which information from the hidden states is propagated to
        the fully-connected classification layers: the state from the last
        step, the max element across all steps, the mean of all steps, or
        all three preceding possibilities concatenated together.
    `lr_scheduler`: in (None, 'cyclic', 'plateau'), optional
        Toggles the use of a learning rate scheduler, either 'cyclic'
        (CyclicLR), 'plateau' (ReduceLROnPlateau), or None (constant lr).
    `lr_scheduler_args`: None or dict, optional
        Optional dictionary of arguments to be passed to the lr_scheduler.
        Overwrites every arg of the scheduler, apart from the optimizer.
    `lr_scheduler_interval`: 'step' or 'epoch', default = 'step'
        Interval at which .step() is called for the lr_scheduler.
    `lr_scheduler_monitor`: str, default = 'train_batch_loss
        Metric monitored by scheduler, if applicable (e.g. ReduceLROnPlateau).
    `save_models_to_wandb`: bool, default = False
        Toggles saving the best models according to validation accuracy or
        loss to wandb.
    """

    def __init__(
        self,
        seq_len: int,
        lr: float = 1e-3,
        l2_reg: float = 0.0,
        hidden_strategy: Literal["last", "max", "mean", "concat"] = "last",
        lr_scheduler: Literal["cyclic", "plateau", None] = None,
        lr_scheduler_args: Union[Dict[str, float], None] = None,
        lr_scheduler_interval: Literal["step", "epoch"] = "step",
        lr_scheduler_monitor: str = "train_batch_loss",
        save_models_to_wandb: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.save_models_to_wandb = save_models_to_wandb
        self.lr = lr
        self.l2_reg = l2_reg
        self.hidden_strategy = hidden_strategy
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_scheduler_monitor = lr_scheduler_monitor

        # Initializing best val loss and accuracy achieved.
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

        # torchmetrics methods for computing various statistics.
        # Needs to be wrapped in a ModuleDict for proper hook registering
        # and device-moving: https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metrics-and-devices
        metrics_dict = {
            "acc": torchmetrics.Accuracy,
            "precision": torchmetrics.Precision,
            "recall": torchmetrics.Recall,
            "specificity": torchmetrics.Specificity,
            "f1": torchmetrics.F1,
            "auroc": torchmetrics.AUROC,
            "avg_precision": torchmetrics.AveragePrecision,
        }

        self.train_metrics_dict = nn.ModuleDict(
            {"train_" + key: val() for key, val in metrics_dict.items()}
        )
        self.val_metrics_dict = nn.ModuleDict(
            {"val_" + key: val() for key, val in metrics_dict.items()}
        )
        self.test_metrics_dict = nn.ModuleDict(
            {"test_" + key: val() for key, val in metrics_dict.items()}
        )

    def configure_optimizers(self):
        """Adam optimizer with various learning rate scheduler options."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optim_dict = {"optimizer": optimizer}

        scheduler_dict = {
            "cyclic": torch.optim.lr_scheduler.CyclicLR,
            "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        }
        scheduler = scheduler_dict.get(self.lr_scheduler, None)
        if scheduler:
            if self.lr_scheduler_args:
                scheduler = scheduler(optimizer, **self.lr_scheduler_args)
            else:
                scheduler = scheduler(optimizer)
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": self.lr_scheduler_interval,
                "monitor": self.lr_scheduler_monitor,
                "frequency": 1,
            }
        return optim_dict

    def forward(
        self, input: torch.Tensor, hiddens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns outputs as a flattened vector, along with the hidden state."""
        if hasattr(self, "embedding"):
            input = self.embedding(input)
        output, hiddens = self.rnn(input, hiddens)

        if self.hidden_strategy == "last":
            output = output[:, -1]
        elif self.hidden_strategy == "max":
            output = output.max(dim=1).values
        elif self.hidden_strategy == "mean":
            output = output.mean(dim=1)
        elif self.hidden_strategy == "concat":
            output = torch.cat(
                (output[:, -1], output.max(dim=1).values, output.mean(dim=1)), dim=1
            )

        for layer in self.fc_layers:
            output = layer(output)
        return output.view(-1), hiddens

    def scores_loss_hiddens(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        hiddens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the scores, losses, and hiddens, with L2 regularization applied to the weights
        of all fully connected layers, if l2_reg.
        """
        scores, hiddens = self(inputs, hiddens)
        loss = F.binary_cross_entropy_with_logits(scores, targets.float())
        if self.l2_reg:
            assert self.l2_reg > 0, "L2 regularization parameter must be positive."
            for layer in self.fc_layers:
                if isinstance(layer, nn.Linear):
                    loss += self.l2_reg * (layer.weight ** 2).sum()
        return scores, loss, hiddens

    def get_probs_hiddens(
        self, input: torch.Tensor, hiddens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns probs, hiddens, with probs shaped as (batch_size, )."""
        is_training = self.training
        self.eval()
        with torch.no_grad():
            scores, hiddens = self(input, hiddens)
            probs = scores.sigmoid()
        if is_training:
            self.train()
        return probs, hiddens

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        hiddens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        inputs, targets = batch
        scores, loss, hiddens = self.scores_loss_hiddens(
            inputs, targets, hiddens=hiddens
        )
        self.log("train_batch_loss", loss, prog_bar=True)
        for metric in self.train_metrics_dict.values():
            metric(scores.detach(), targets)
        # pl requires the loss be returned via the 'loss' key
        out = {"loss": loss}
        # Returning hiddens in training_step leads to a pl MisconfigurationError
        # if truncated_bptt_steps has not been set.
        if self.truncated_bptt_steps:
            out["hiddens"] = hiddens
        return out

    def training_epoch_end(
        self, training_step_outputs: Dict[str, torch.Tensor]
    ) -> None:
        """If tbptt_split_batch is used, training_step_outputs will contain
        lists of lists of dicts, rather than single lists of dicts, so a
        flattening helper function must be used before computing the mean loss.
        """
        mean_loss = torch.tensor(
            [batch["loss"] for batch in self._flatten_2d_list(training_step_outputs)]
        ).mean()
        self.log("train_epoch_loss", mean_loss, prog_bar=True)
        for name, metric in self.train_metrics_dict.items():
            self.log(name, metric, prog_bar=True)

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        inputs, targets = batch
        scores, loss, _ = self.scores_loss_hiddens(inputs, targets)
        for metric in self.val_metrics_dict.values():
            metric(scores.detach(), targets)
        return {"val_loss": loss}

    def validation_epoch_end(
        self, validation_step_outputs: Dict[str, torch.Tensor]
    ) -> None:
        """If tbptt_split_batch is used, training_step_outputs will contain
        lists of lists of dicts, rather than single lists of dicts, so a
        flattening helper function must be used before computing the mean loss.
        """
        self._curr_val_loss = torch.tensor(
            [
                batch["val_loss"]
                for batch in self._flatten_2d_list(validation_step_outputs)
            ]
        ).mean()
        self.log("val_loss", self._curr_val_loss, prog_bar=True)
        for name, metric in self.val_metrics_dict.items():
            self.log(name, metric, prog_bar=True)

        # Update and log the best val acc and loss seen so far.
        # booleans used if save_best_model
        self._curr_val_acc = self.val_metrics_dict["val_acc"].compute()
        is_best_val_loss = self._curr_val_loss < self.best_val_loss
        is_best_val_acc = self._curr_val_acc > self.best_val_acc
        self.best_val_loss = min(self.best_val_loss, self._curr_val_loss)
        self.best_val_acc = max(self.best_val_acc, self._curr_val_acc)
        self.log("best_val_acc", self.best_val_acc, prog_bar=True)
        self.log("best_val_loss", self.best_val_loss, prog_bar=True)

        if is_best_val_acc and self.save_models_to_wandb:
            self.save_best_model(metric="val_acc")
        if is_best_val_loss and self.save_models_to_wandb:
            self.save_best_model(metric="val_loss")

    def save_best_model(self, metric: str) -> None:
        """Save best models (state_dict) by metric to .pt format."""
        model_name = "model_best_" + metric
        model_json_name = model_name + "_hparam_dict.json"
        model_state_dict_name = model_name + ".pt"
        model_dict = {
            **{
                "val_loss": self._curr_val_loss.item(),
                "val_acc": self._curr_val_acc.item(),
            },
            **self.hparams,
        }
        model_dict_json = json.dumps(model_dict)
        with open(model_json_name, "w") as outfile:
            outfile.write(model_dict_json)
        wandb.save(model_json_name)
        torch.save(self.state_dict(), model_state_dict_name)
        wandb.save(model_state_dict_name)

        print(
            f"Saved best {metric} at global step: {self.global_step}",
            f"Epoch: {self.current_epoch}",
            f"Validation accuracy: {self.val_metrics_dict['val_acc'].compute().item()}",
            f"Validation Loss: {self._curr_val_loss.item()}",
            sep="\n",
        )

    def test_step(self):
        raise NotImplementedError("Should not be using test dataset yet.")

    def test_epoch_end(self):
        raise NotImplementedError("Should not be using test dataset yet.")

    def tbptt_split_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor], split_size: Union[int, float]
    ) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        """Passing a float to split_size chooses the split via a fraction of
        the sequence length. If used, by setting truncated_bptt_steps to not
        None, the shape of the x_step_outputs that x_step passes to
        x_epoch_end is altered from a list of dictionaries (of len determined
        by batch size) to a list (of len determined by batch size) of lists
        (determined by the number of tbptt splits) of dictionaries.
        """
        if type(split_size) == float:
            assert 0 <= split_size <= 1, "split_size must be between 0 and 1."
            split_size = int(split_size * self.seq_len)

        inputs, targets = batch
        time_dims = [x.shape[0] for x in inputs]
        assert len(time_dims) >= 1, "Unable to determine batch time dimension."
        assert all(
            x == time_dims[0] for x in time_dims
        ), "Batch time dimension length is ambiguous."

        splits = []
        for t in range(0, time_dims[0], split_size):
            split_text = inputs[:, t : t + split_size]
            split_targets = targets[:, t : t + split_size]
            splits.append((split_text, split_targets))
        return splits

    def _flatten_2d_list(self, list2d):
        """Helper function for recursively flattening 2D lists. General recursive approach avoided,
        as it can hit recursion limit."""
        flat_list = []
        for item in list2d:
            if isinstance(item, list):
                for sub_item in item:
                    flat_list.append(sub_item)
            else:
                flat_list.append(item)
        return flat_list


class LitOneHotCharRNNAV(LitRNNLoggingBaseAV):
    """RNN architectures for arxiv/vixra classification using
    character-level, one-hot encoded text.

    Description
    ----------
    Must specify `rnn_type` which can be any of `RNN`, `LSTM` or `GRU`.
    Test-data methods are currently not implemented to prevent early peeking.

    Args
    ----------
    `seq_len`: int
        Sequence length; standard torch rnn arg.
    `tokens`: str or pd.DataFrame
        DataFrame object or path to character feather file. Used to determine
        input_size arg of recurrent layer.
    `rnn_type`: in ('RNN', 'LSTM', 'GRU')
        RNN architecture.
    `num_layers`: int, default = 1
        Number of rnn layers; standard torch rnn arg.
    `hidden_size`: int, default = 128
        Hidden dimension size; standard torch rnn arg.
    `hidden_strategy`: in ('last', 'max', 'mean', 'concat'), default = 'last'
        Determines which information from the hidden states is propagated to
        the fully-connected classification layers: the state from the last step,
        the max element across all steps, the mean of all steps, or all three
        preceding possibilities concatenated together.
    `bidirectional`: bool, default = False
        Flag for whether the architecture is bidirectional
    `lr`: float, default = 1e-3
        Learning rate for Adam optimizer.
    `lr_scheduler`: in (None, 'cyclic', 'plateau'), optional
        Toggles the use of a learning rate scheduler, either 'cyclic'
        (CyclicLR), 'plateau' (ReduceLROnPlateau), or None (constant lr).
    `lr_scheduler_args`: None or dict, optional
        Optional dictionary of arguments to be passed to the lr_scheduler.
        Overwrites every arg of the scheduler, apart from the optimizer.
    `lr_scheduler_interval`: 'step' or 'epoch', default = 'step'
        Interval at which .step() is called for the lr_scheduler.
    `lr_scheduler_monitor`: str, default = 'train_batch_loss
        Metric monitored by scheduler, if applicable (e.g. ReduceLROnPlateau).
    `l2_reg`: float, default = 0.
        Parameter controlling strength of l2 regularization on fully-connected
        layer weights.
    `fc_dims`: None or Sequence[int], optional
        Dimensions of any hidden layers between rnn output and class prediction
        layer.
    `fc_dropout`: None or float, optional
        Set dropout probability before fully-connected layers.
    `zero_fc_bias_init`: bool, default = True
        Flag for initializing the bias of all fully connected layers to zero.
    `recurrent_dropout`: None or float, optional
        Set dropout probability between recurrent layers, if any.
    `truncated_bptt_steps`: None or int or float, optional
        Implements truncated backpropagation through time, if provided.
    `save_models_to_wandb`: bool, default = False
        Toggles saving the best models according to validation accuracy or
        loss to wandb.
    `logging_kwargs`: None or dict, optional
        Optional kwargs which don't affect performance, but which will be
        tracked by loggers such as wandb. Useful for tracking batch size, e.g.
    """

    def __init__(
        self,
        seq_len: int,
        tokens: Union[str, pd.DataFrame],
        rnn_type: Literal["RNN", "LSTM", "GRU"],
        num_layers: int = 1,
        hidden_size: int = 128,
        hidden_strategy: Literal["last", "max", "mean", "concat"] = "last",
        bidirectional: bool = False,
        lr: float = 1e-3,
        lr_scheduler: Literal["cyclic", "plateau", None] = None,
        lr_scheduler_args: Union[Dict[str, float], None] = None,
        lr_scheduler_interval: Literal["step", "epoch"] = "step",
        lr_scheduler_monitor: str = "train_batch_loss",
        l2_reg: float = 0.0,
        fc_dims: Optional[Sequence[int]] = None,
        fc_dropout: Optional[float] = None,
        zero_fc_bias_init: bool = True,
        recurrent_dropout: Optional[float] = None,
        truncated_bptt_steps: Optional[int] = None,
        save_models_to_wandb: bool = False,
        **logging_kwargs: Dict[str, Union[float, int, str]],
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            lr=lr,
            l2_reg=l2_reg,
            hidden_strategy=hidden_strategy,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_monitor=lr_scheduler_monitor,
            save_models_to_wandb=save_models_to_wandb,
        )

        # Logging hyperparameters to the hparams attr of the class (pl feature)
        # saves all args of __init__ to self.hparam.  So, can get the lr via
        # self.hparams['lr'], for instance.
        self.save_hyperparameters(ignore="tokens")

        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = deepcopy(tokens)
        self.hparams["input_size"] = len(self.tokens_df)

        # The fully connected layer size depends on the bidirectional flag
        # and whether or not the 'concat' hidden_strategy is used.
        self._fc_input_size = (
            (2 if bidirectional else 1)
            * (3 if hidden_strategy == "concat" else 1)
            * hidden_size
        )
        # Need to set truncated_bptt_steps as an attr to implement tbptt.
        if truncated_bptt_steps is not None:
            self.truncated_bptt_steps = truncated_bptt_steps

        rnn_dict = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        self.rnn = rnn_dict[rnn_type]
        self._rnn_dict = {
            "input_size": self.hparams["input_size"],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "bidirectional": bidirectional,
        }
        if recurrent_dropout is not None:
            self._rnn_dict["dropout"] = None
        self.rnn = self.rnn(**self._rnn_dict)
        if self.hparams["fc_dims"] is None:
            self.hparams["fc_dims"] = []
        in_dims = [self._fc_input_size] + self.hparams["fc_dims"]
        out_dims = self.hparams["fc_dims"] + [1]
        in_out_dims = zip(in_dims, out_dims)
        self.linears = [nn.Linear(i, o) for i, o in in_out_dims]
        if zero_fc_bias_init:
            for layer in self.linears:
                nn.init.constant_(layer.bias.data, 0.0)

        # Insert Dropout before (if fc_dropout) and ReLUs between FC layers.
        if self.hparams["fc_dropout"]:
            num_layers = 3 * len(self.linears) - 1
            num_linears = len(self.linears)
            num_dropouts = len(self.linears)
            num_relus = num_layers - num_linears - num_dropouts
            self.fc_layers = [None for _ in range(num_layers)]
            self.fc_layers[::3] = [
                nn.Dropout(self.hparams["fc_dropout"]) for _ in range(num_dropouts)
            ]
            self.fc_layers[1::3] = self.linears
            self.fc_layers[2::3] = [nn.ReLU() for _ in range(num_relus)]
            self.fc_layers = nn.ModuleList(self.fc_layers)
        else:
            num_layers = 2 * len(self.linears) - 1
            num_linears = len(self.linears)
            num_relus = num_layers - num_linears
            self.fc_layers = [None for _ in range(num_layers)]
            self.fc_layers[::2] = self.linears
            self.fc_layers[1::2] = [nn.ReLU() for _ in range(num_relus)]
            self.fc_layers = nn.ModuleList(self.fc_layers)


class LitEmbeddingRNNAV(LitRNNLoggingBaseAV):
    """RNN architectures for arxiv/vixra classification using
    word-level tokenized text.

    Description
    ----------
    Must specify `rnn_type` which can be any of `RNN`, `LSTM` or `GRU`. Trained
    on text which had .strip, unidecode, and .lower applied to it. Test-data
    methods are currently not implemented to prevent early peeking. tokens
    DataFrame (or associated feather file) is expected to have a 'count' column
    tallying the number of times each word appeared in the training set.

    Args
    ----------
    `seq_len`: int
        Sequence length; standard torch rnn arg.
    `tokens`: str or pd.DataFrame
        DataFrame object or path to vocabulary feather file. Used to determine
        num_embeddings arg of nn.Embedding.
    `min_word_count`: int, default = 1
        Minimum count for a word in tokens to be included in the vocabulary.
    `rnn_type`: in ('RNN', 'LSTM', 'GRU')
        RNN architecture.
    `embedding_dim`: int, default = 256
        Size of embedding dimension; standard torch nn.Embedding arg.
    `num_layers`: int, default = 1
        Number of rnn layers; standard torch rnn arg.
    `hidden_size`: int, default = 128
        Hidden dimension size; standard torch rnn arg.
    `hidden_strategy`: in ('last', 'max', 'mean', 'concat'), default = 'last'
        Determines which information from the hidden states is propagated to
        the fully-connected classification layers: the state from the last step,
        the max element across all steps, the mean of all steps, or all three
        preceding possibilities concatenated together.
    `bidirectional`: bool, default = False
        Flag for whether the architecture is bidirectional
    `lr`: float, default = 1e-3
        Learning rate for Adam optimizer.
    `lr_scheduler`: in ('cyclic', 'plateau', None), optional
        Toggles the use of a learning rate scheduler, either 'cyclic'
        (CyclicLR), 'plateau' (ReduceLROnPlateau), or None (constant lr).
    `lr_scheduler_args`: None or dict, optional
        Optional dictionary of arguments to be passed to the lr_scheduler.
        Overwrites every arg of the scheduler, apart from the optimizer.
    `lr_scheduler_interval`: 'step' or 'epoch', default = 'step'
        Interval at which .step() is called for the lr_scheduler.
    `lr_scheduler_monitor`: str, default = 'train_batch_loss
        Metric monitored by scheduler, if applicable (e.g. ReduceLROnPlateau).
    `l2_reg`: float, default = 0.
        Parameter controlling strength of l2 regularization on final
        fully-connected layer weights.
    `fc_dims`: None or Sequence[int], optional
        Dimensions of any hidden layers between rnn output and class prediction
        layer.
    `fc_dropout`: None or float, optional
        Set dropout probability before fully-connected layers.
    `zero_fc_bias_init`: bool, default = True
        Flag for initializing the bias of all fully connected layers to zero.
    `recurrent_dropout`: None or float, optional
        Set dropout probability between recurrent layers, if any.
    `truncated_bptt_steps`: None or int or float, optional
        Implements truncated backpropagation through time, if provided.
    `save_models_to_wandb`: bool, default = False
        Toggles saving the best models according to validation accuracy and
        loss to wandb.
    `logging_kwargs`: None or dict, optional
        Optional kwargs which don't affect performance, but which will be
        tracked by loggers such as wandb. Useful for tracking batch size, e.g.
    `embedding_from_pretrained`: None or torch.Tensor, optional
        Optionally load a pre-trained tensor into the embedding layer.
    `freeze_pretrained`: bool, default = True
        Flag for freezing pretrained embedding layer.

    Notable Methods
    ----------
    `get_embedding_weights`:
        Returns the embedding layer weights.
    `get_embedding_vector`:
        Returns the vector corresponding to a given word index.
    `get_unit_embedding_vector`:
        Returns the vector corresponding to a given word index normalized to
        length 1.
    `get_embedding_cosine`:
        Returns the cosine between the vectors corresponding to a pair of word
        indices.
    `get_embedding_dot`:
        Returns the dot-product between the vectors corresponding to a pair of
        word indices.
    """

    def __init__(
        self,
        seq_len: int,
        tokens: Union[str, pd.DataFrame],
        rnn_type: Literal["RNN", "LSTM", "GRU"],
        min_word_count: int = 1,
        embedding_dim: int = 256,
        num_layers: int = 1,
        hidden_size: int = 128,
        hidden_strategy: Literal["last", "max", "mean", "concat"] = "last",
        bidirectional: bool = False,
        lr: float = 1e-3,
        lr_scheduler: Literal["cyclic", "plateau", None] = None,
        lr_scheduler_args: Union[Dict[str, float], None] = None,
        lr_scheduler_interval: Literal["step", "epoch"] = "step",
        lr_scheduler_monitor: str = "train_batch_loss",
        l2_reg: float = 0.0,
        fc_dims: Optional[Sequence[int]] = None,
        fc_dropout: Optional[float] = None,
        zero_fc_bias_init: bool = True,
        recurrent_dropout: Optional[float] = None,
        truncated_bptt_steps: Optional[int] = None,
        save_models_to_wandb: bool = False,
        embedding_from_pretrained: Optional[torch.Tensor] = None,
        freeze_pretrained: Optional[bool] = True,
        **logging_kwargs: Dict[str, Union[float, int, str]],
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            lr=lr,
            l2_reg=l2_reg,
            hidden_strategy=hidden_strategy,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_monitor=lr_scheduler_monitor,
            save_models_to_wandb=save_models_to_wandb,
        )

        # Logging hyperparameters to the hparams attr of the class (pl feature)
        # saves all args of __init__ to self.hparam.  So, can get the lr via
        # self.hparams['lr'], for instance.
        self.save_hyperparameters(ignore=("tokens", "embedding_from_pretrained"))
        PAD_IDX = 0

        # Get num_embeddings from tokens
        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = deepcopy(tokens)
        if min_word_count > 1:
            if "count" in self.tokens_df:
                self.tokens_df = self.tokens_df[
                    self.tokens_df["count"] >= min_word_count
                ]
            else:
                warnings.warn(
                    "count column does not exist in tokens DataFrame, min_word_count arg ignored."
                )
        # + 2 for padding and <UNK>.
        self.hparams["num_embeddings"] = len(self.tokens_df) + 2

        # The fully connected layer size depends on the bidirectional flag
        # and whether or not the 'concat' hidden_strategy is used.
        self._fc_input_size = (
            (2 if bidirectional else 1)
            * (3 if hidden_strategy == "concat" else 1)
            * hidden_size
        )
        # Need to set truncated_bptt_steps as an attr to implement tbptt.
        if truncated_bptt_steps is not None:
            self.truncated_bptt_steps = truncated_bptt_steps

        # Embedding layer with 0 as padding idx.
        if embedding_from_pretrained is not None:
            sliced_pretrained_embedding = embedding_from_pretrained[
                : self.hparams["num_embeddings"]
            ]
            self.embedding = nn.Embedding.from_pretrained(
                sliced_pretrained_embedding,
                freeze=freeze_pretrained,
                padding_idx=PAD_IDX,
            )
        else:
            self.embedding = nn.Embedding(
                num_embeddings=self.hparams["num_embeddings"],
                embedding_dim=embedding_dim,
                padding_idx=PAD_IDX,
            )

        rnn_dict = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        self.rnn = rnn_dict[rnn_type]
        self._rnn_dict = {
            "input_size": embedding_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "bidirectional": bidirectional,
        }
        if recurrent_dropout is not None:
            self._rnn_dict["dropout"] = recurrent_dropout
        self.rnn = self.rnn(**self._rnn_dict)

        if self.hparams["fc_dims"] is None:
            self.hparams["fc_dims"] = []
        in_dims = [self._fc_input_size] + self.hparams["fc_dims"]
        out_dims = self.hparams["fc_dims"] + [1]
        in_out_dims = zip(in_dims, out_dims)
        self.linears = [nn.Linear(i, o) for i, o in in_out_dims]
        if zero_fc_bias_init:
            for layer in self.linears:
                nn.init.constant_(layer.bias.data, 0.0)

        # Insert Dropout before (if fc_dropout) and ReLUs between FC layers.
        if self.hparams["fc_dropout"]:
            num_layers = 3 * len(self.linears) - 1
            num_linears = len(self.linears)
            num_dropouts = len(self.linears)
            num_relus = num_layers - num_linears - num_dropouts
            self.fc_layers = [None for _ in range(num_layers)]
            self.fc_layers[::3] = [
                nn.Dropout(self.hparams["fc_dropout"]) for _ in range(num_dropouts)
            ]
            self.fc_layers[1::3] = self.linears
            self.fc_layers[2::3] = [nn.ReLU() for _ in range(num_relus)]
            self.fc_layers = nn.ModuleList(self.fc_layers)
        else:
            num_layers = 2 * len(self.linears) - 1
            num_linears = len(self.linears)
            num_relus = num_layers - num_linears
            self.fc_layers = [None for _ in range(num_layers)]
            self.fc_layers[::2] = self.linears
            self.fc_layers[1::2] = [nn.ReLU() for _ in range(num_relus)]
            self.fc_layers = nn.ModuleList(self.fc_layers)

    def get_embedding_weights(self) -> torch.Tensor:
        """Returns the mean of the word and context embedding weights."""
        with torch.no_grad():
            embedding_weights = self.embedding.weight.data
            return embedding_weights.clone().detach()

    def get_embedding_vector(self, idx: int) -> torch.Tensor:
        """
        Returns the mean of the word and context vectors corresponding to a
        given index.
        """
        with torch.no_grad():
            idx_t = torch.tensor(idx, device=self.device)
            word_vector = self.embedding(idx_t)
            return word_vector.clone().detach()

    def get_unit_embedding_vector(self, idx: int) -> torch.Tensor:
        """
        Returns the mean of the word and context vectors with corresponding to
        a given index whose length has been normalized to 1.
        """
        vector = self.get_embedding_vector(idx)
        vector_norm = (vector @ vector).sqrt()
        unit_vector = vector / vector_norm
        return unit_vector.clone().detach()

    def get_embedding_cosine(self, idx1: int, idx2: int) -> torch.Tensor:
        """Returns the cosine between the mean vectors corresponding to the
        provided indices.
        """
        with torch.no_grad():
            unit_vector_1 = self.get_unit_embedding_vector(idx1)
            unit_vector_2 = self.get_unit_embedding_vector(idx2)
            cos = unit_vector_1 @ unit_vector_2
            return cos.clone().detach()

    def get_embedding_dot(self, idx1: int, idx2: int) -> torch.Tensor:
        """Returns the dot product between the mean vectors corresponding
        to the provided indices.
        """
        with torch.no_grad():
            vector_1 = self.get_embedding_vector(idx1)
            vector_2 = self.get_embedding_vector(idx2)
            dot = vector_1 @ vector_2
            return dot.clone().detach()