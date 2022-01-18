import json
from typing import Dict, Tuple, Sequence, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchmetrics
import wandb


class LitMinimalLoggingBase(pl.LightningModule):
    """Minimal PyTorch Lightning base class defining common methods
    for train, validation, and test steps. Uses wandb logging.

    Description
    ----------

    Must overwrite the following methods to use:
    - forward
    - scores_loss
    - configure optimizers

    Args
    ----------
    seq_len : int
        Length of sequences passed to the recurrent architecture.
    `save_models_to_wandb`: bool, default False
        Toggles saving the best models according to validation accuracy or
        loss to wandb.
    """

    def __init__(self, seq_len: int, save_models_to_wandb: bool = False) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.save_models_to_wandb = save_models_to_wandb
        # Attrs to store the best val_acc and val_loss seen at the end of any
        # validation epoch
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

    def forward(self, input: Tensor) -> Tensor:
        """
        Overwrite. Expected to return outputs tensor.
        """
        raise NotImplementedError("Must overwrite the forward method.")

    def scores_loss(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Overwrite. Expected to return (scores, losses) tuple.
        """
        raise NotImplementedError("Must overwrite the scores_loss method.")

    def configure_optimizers(self):
        """Overwrite."""
        raise NotImplementedError("Must overwrite the configure_optimizers method.")

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        inputs, targets = batch
        scores, loss = self.scores_loss(inputs, targets)
        for metric in self.train_metrics_dict.values():
            metric(scores.detach(), targets)

        return {"loss": loss}

    def training_epoch_end(
        self, training_step_outputs: Sequence[Dict[str, Tensor]]
    ) -> None:
        mean_loss = torch.tensor(
            [batch["loss"] for batch in training_step_outputs]
        ).mean()
        self.log("train_epoch_loss", mean_loss, prog_bar=True)
        for name, metric in self.train_metrics_dict.items():
            self.log(name, metric, prog_bar=True)

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        inputs, targets = batch
        scores, loss = self.scores_loss(inputs, targets)
        for metric in self.val_metrics_dict.values():
            metric(scores.detach(), targets)
        return {"val_loss": loss}

    def validation_epoch_end(
        self, validation_step_outputs: Sequence[Dict[str, Tensor]]
    ) -> None:
        self._curr_val_loss = torch.tensor(
            [batch["val_loss"] for batch in validation_step_outputs]
        ).mean()
        self._curr_val_acc = self.val_metrics_dict["val_acc"].compute()
        self.log("val_loss", self._curr_val_loss, prog_bar=True)
        for name, metric in self.val_metrics_dict.items():
            self.log(name, metric, prog_bar=True)

        # Update and log the best val acc and loss seen so far.
        # booleans used if save_best_model.
        is_best_val_loss = self._curr_val_loss < self.best_val_loss
        is_best_val_acc = self._curr_val_acc > self.best_val_acc
        self.best_val_loss = min(self.best_val_loss, self._curr_val_loss)
        self.best_val_acc = max(self.best_val_acc, self._curr_val_acc)
        self.log("best_val_acc", self.best_val_acc, prog_bar=True)
        self.log("best_val_loss", self.best_val_loss, prog_bar=True)

        # Save best models, if desired/implementd.
        if is_best_val_acc and self.save_models_to_wandb:
            self.save_best_model(metric="val_acc")
        if is_best_val_loss and self.save_models_to_wandb:
            self.save_best_model(metric="val_loss")

    def save_best_model(self, metric: str) -> None:
        """Overwrite.

        Save the state_dict to wandb based on metric,
        which is either 'val_loss' or 'val_acc'.
        """
        raise NotImplementedError("Must overwrite the save_best_model method.")

    def test_step(self):
        raise NotImplementedError("Should not be using test dataset yet.")

    def test_epoch_end(self):
        raise NotImplementedError("Should not be using test dataset yet.")


class LitOneHotFC(LitMinimalLoggingBase):
    """Extremely minimal fully connected architecture with for one-hot encoded
    text.

    Description
    ----------
    Takes in the one-hot tensors and flattens them on the fly.
    Layers specified by passing a tuple of hidden dimensions.
    Test-data methods are currently not implemented to prevent early peeking.

    Args
    ----------
    seq_len : int
        Length of text (counted by number of characters.)
    tokens : str or pd.DataFrame
        DataFrame object or path to character feather file, determines the size
        of the one-hot vectors.
    `fc_dims`: Sequence[int], default None
        Tuple of hidden dimension sizes. Default implements [64, 32].
    `zero_fc_bias_init`: bool, default True
        Flag for initializing the bias of all fully connected layers to zero.
    `lr`: float, default 1e-3
        Learning rate for Adam optimizer.
    `nonlinearity`: bool, default True
        Boolean for whether to use insert ReLUs between hidden layers. A flag to
        test linear models.
    `save_models_to_wandb`: bool, default False
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
        fc_dims: Sequence[int] = None,
        zero_fc_bias_init: bool = True,
        lr: float = 1e-3,
        nonlinearity: bool = True,
        save_models_to_wandb: bool = False,
        **logging_kwargs: Dict[str, Union[float, int, str]],
    ) -> None:
        super().__init__(seq_len, save_models_to_wandb)
        # Logging hyperparameters to the hparams attr of the class (pl feature)
        # saves all args of __init__ to self.hparam.  So, can get the lr via
        # self.hparams['lr'], for instance.
        self.save_hyperparameters(ignore="tokens")

        # Get input_size from tokens file.
        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = tokens
        self.hparams["input_size"] = len(self.tokens_df)

        if not fc_dims:
            self.fc_dims = []
        else:
            self.fc_dims = fc_dims
        # Zip in and out dimensions together
        flattened_size = seq_len * self.hparams["input_size"]
        in_dims = [flattened_size] + self.fc_dims
        out_dims = self.fc_dims + [1]
        in_out_dims = zip(in_dims, out_dims)
        self.fc_layers = [nn.Linear(i, o) for i, o in in_out_dims]
        if zero_fc_bias_init:
            for layer in self.fc_layers:
                nn.init.constant_(layer.bias.data, 0.0)

        # Insert ReLUs between all FC layers, if nonlinearity.
        if nonlinearity:
            self.layers = [None for _ in range(2 * len(self.fc_layers) - 1)]
            self.layers[::2] = self.fc_layers
            self.layers[1::2] = [nn.ReLU() for _ in range(len(self.fc_layers) - 1)]
            self.layers = nn.ModuleList(self.layers)
        else:
            self.layers = nn.ModuleList(self.fc_layers)

    def forward(self, input: Tensor) -> Tensor:
        # Flatten and pass through layers.
        seq_len, input_size = self.hparams["seq_len"], self.hparams["input_size"]
        output = input.view(-1, seq_len * input_size)
        output = (output - output.mean()) / output.std()
        for layer in self.layers:
            output = layer(output)
        return output.view(-1)

    def scores_loss(self, inputs: Tensor, targets: Tensor) -> Tuple[Tensor]:
        scores = self(inputs)
        loss = F.binary_cross_entropy_with_logits(scores, targets.float())
        return scores, loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def save_best_model(self, metric: str) -> None:
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

    def get_probs(self, input: Tensor) -> Tensor:
        """Returns the probability predictions."""
        is_training = self.training
        self.eval()
        with torch.no_grad():
            probs = self(input).sigmoid()
        if is_training:
            self.train()
        return probs
