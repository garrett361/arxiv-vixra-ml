from copy import deepcopy
from typing import Dict, Tuple, Sequence, Optional, Union
from typing_extensions import Literal
import warnings

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics

from .one_hot_utils import str_to_one_hot
from .simple_recurrent import LitRNNLoggingBaseAV


class LitRNNLoggingBaseLM(LitRNNLoggingBaseAV):
    """Custom PyTorch Lightning base class for recurrent architectures
    for language-model generation with logging. Defining common methods
    for train, validation, and test steps.

    Description
    ----------
    Written such that one-hot and embedding models need only subclass
    LitRNNLoggingBaseLM with __init__ method overwritten.  Expects
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
        lr_scheduler: Literal["cyclic", "plateau", None] = None,
        lr_scheduler_args: Union[Dict[str, float], None] = None,
        lr_scheduler_interval: Literal["step", "epoch"] = "step",
        lr_scheduler_monitor: str = "train_batch_loss",
        save_models_to_wandb: bool = False,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_monitor=lr_scheduler_monitor,
            save_models_to_wandb=save_models_to_wandb,
        )
        # The mdmc_reduce parameter needs to be included for some torchmetrics
        # due to the use of multiclass, multidimensional outputs.
        metrics_dict = {
            "acc": torchmetrics.Accuracy,
            "precision": torchmetrics.Precision,
            "recall": torchmetrics.Recall,
            "specificity": torchmetrics.Specificity,
        }

        self.train_metrics_dict = nn.ModuleDict(
            {
                "train_" + key: val(mdmc_average="global")
                for key, val in metrics_dict.items()
            }
        )
        self.val_metrics_dict = nn.ModuleDict(
            {
                "val_" + key: val(mdmc_average="global")
                for key, val in metrics_dict.items()
            }
        )
        self.test_metrics_dict = nn.ModuleDict(
            {
                "test_" + key: val(mdmc_average="global")
                for key, val in metrics_dict.items()
            }
        )

    def forward(
        self, input: torch.Tensor, hiddens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns scores shaped as (batch_size, classes, seq_len) and
        the hidden state. This output shape is expected both by cross_entropy
        and the torchmetrics classes.
        """
        if hasattr(self, "embedding"):
            input = self.embedding(input)
        output, hiddens = self.rnn(input, hiddens)

        for layer in self.fc_layers:
            output = layer(output)
        scores = output.permute(0, 2, 1)
        return scores, hiddens

    def scores_loss_hiddens(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        hiddens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the scores, losses, and hiddens."""
        scores, hiddens = self(inputs, hiddens)
        loss = F.cross_entropy(scores, targets)
        return scores, loss, hiddens

    def get_probs_hiddens(
        self, input: torch.Tensor, hiddens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns probs, hiddens, with probs shaped as
        (batch_size, classes, seq_len).
        """
        is_training = self.training
        self.eval()
        with torch.no_grad():
            scores, hiddens = self(input, hiddens)
            probs = scores.softmax(dim=1)
        if is_training:
            self.train()
        return probs, hiddens


class LitOneHotCharRNNNextLM(LitRNNLoggingBaseLM):
    """RNN architectures for language-model creation using
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
        lr: float = 1e-3,
        lr_scheduler: Literal["cyclic", "plateau", None] = None,
        lr_scheduler_args: Union[Dict[str, float], None] = None,
        lr_scheduler_interval: Literal["step", "epoch"] = "step",
        lr_scheduler_monitor: str = "train_batch_loss",
        fc_dims: Optional[Sequence[int]] = None,
        fc_dropout: Optional[float] = None,
        zero_fc_bias_init: bool = True,
        recurrent_dropout: Optional[float] = None,
        truncated_bptt_steps: Optional[int] = None,
        save_models_to_wandb: bool = False,
        **logging_kwargs: Dict[str, Union[float, int, str]]
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            lr=lr,
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
        if truncated_bptt_steps is not None:
            self.truncated_bptt_steps = truncated_bptt_steps

        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = deepcopy(tokens)
        self.hparams["input_size"] = len(self.tokens_df)
        self._char_to_idx = {
            row["char"]: row["idx"] for _, row in self.tokens_df.iterrows()
        }
        self._idx_to_char = {idx: char for char, idx in self._char_to_idx.items()}

        rnn_dict = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        self.rnn = rnn_dict[rnn_type]
        self._rnn_dict = {
            "input_size": self.hparams["input_size"],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "bidirectional": False,
        }
        if recurrent_dropout is not None:
            self._rnn_dict["dropout"] = recurrent_dropout
        self.rnn = self.rnn(**self._rnn_dict)

        if self.hparams["fc_dims"] is None:
            self.hparams["fc_dims"] = []
        in_dims = [hidden_size] + self.hparams["fc_dims"]
        out_dims = self.hparams["fc_dims"] + [self.hparams["input_size"]]
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

    def _str_to_one_hot(self, s: str) -> torch.Tensor:
        """Returns one-hot-encoded text with a leading singleton dimension."""
        return str_to_one_hot(s=s, char_to_idx=self._char_to_idx)[None].to(self.device)

    def generate_text(self, seed=" ", text_char_len=512, topk=3) -> str:
        """Generates text of character length text_char_len starting from
        the seed str. Randomly chooses the next character in the
        sequence by sampling the topk predictions according to their
        probabilities.
        """
        generated_text = list(seed)
        _, hidden = self.get_probs_hiddens(self._str_to_one_hot(seed))
        while len(generated_text) < text_char_len:
            last_char = generated_text[-1]
            last_char_t = self._str_to_one_hot(last_char)
            probs, hidden = self.get_probs_hiddens(last_char_t, hidden)
            prob_topk = probs.squeeze().topk(dim=0, k=topk)
            rand_idx = torch.multinomial(prob_topk.values, 1)
            next_char_idx = prob_topk.indices[rand_idx].item()
            next_char = self._idx_to_char[next_char_idx]
            generated_text.append(next_char)
        return "".join(generated_text)


class LitEmbeddingRNNNextLM(LitRNNLoggingBaseLM):
    """RNN architectures for language-model creation using
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
        **logging_kwargs: Dict[str, Union[float, int, str]]
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            save_models_to_wandb=save_models_to_wandb,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_monitor=lr_scheduler_monitor,
        )

        # Logging hyperparameters to the hparams attr of the class (pl feature)
        # saves all args of __init__ to self.hparam.  So, can get the lr via
        # self.hparams['lr'], for instance.
        self.save_hyperparameters(ignore=("tokens", "embedding_from_pretrained"))
        if truncated_bptt_steps is not None:
            self.truncated_bptt_steps = truncated_bptt_steps
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
            "bidirectional": False,
        }
        if recurrent_dropout is not None:
            self._rnn_dict["dropout"] = recurrent_dropout
        self.rnn = self.rnn(**self._rnn_dict)

        if self.hparams["fc_dims"] is None:
            self.hparams["fc_dims"] = []
        in_dims = [hidden_size] + self.hparams["fc_dims"]
        out_dims = self.hparams["fc_dims"] + [self.hparams["num_embeddings"]]
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
