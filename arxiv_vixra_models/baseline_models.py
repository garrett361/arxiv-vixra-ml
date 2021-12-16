import json
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
from typing import Dict, Tuple, Sequence, Optional, Union
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
    `seq_len`: int
        Length of sequences passed to the recurrent architecture.
    `save_models_to_wandb`: bool, optional, default = False
        Toggles saving the best models according to validation accuracy or
        loss to wandb.
    """

    def __init__(self,
                 seq_len: int,
                 save_models_to_wandb: Optional[bool] = False) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.save_models_to_wandb = save_models_to_wandb
        # Attrs to store the best val_acc and val_loss seen at the end of any
        # validation epoch
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.

        # torchmetrics methods for computing various statistics.
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.train_auc_roc = torchmetrics.AUROC()
        self.val_auc_roc = torchmetrics.AUROC()
        self.test_auc_roc = torchmetrics.AUROC()

        self.train_average_precision = torchmetrics.AveragePrecision()
        self.val_average_precision = torchmetrics.AveragePrecision()
        self.test_average_precision = torchmetrics.AveragePrecision()

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        Overwrite. Expected to return outputs tensor.
        """
        raise NotImplementedError("Must overwrite the forward method.")

    def scores_loss(self,
                    inputs: torch.Tensor,
                    targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overwrite. Expected to return (scores, losses) tuple.
        """
        raise NotImplementedError(
            "Must overwrite the scores_loss method.")

    def inference(self,
                  input: torch.Tensor) -> torch.Tensor:
        """Inference mode with eval set and no_grad context included. Returns
        probs.
        """
        self.eval()
        with torch.no_grad():
            return self(input).sigmoid()

    def configure_optimizers(self):
        """Overwrite.
        """
        raise NotImplementedError(
            "Must overwrite the configure_optimizers method.")

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        scores, loss = self.scores_loss(x, y)
        self.train_acc(scores, y)
        # Need to detach scores when passing to torchmetrics.AUROC or torchmetrics.AveragePrecision objects
        # to avoid deepcopy error.
        self.train_auc_roc(scores.detach(), y)
        self.train_average_precision(scores.detach(), y)
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs: Sequence[Dict[str, torch.Tensor]]) -> None:
        mean_loss = torch.tensor(
            [batch['loss'] for batch in training_step_outputs]).mean()
        self.log('train_loss', mean_loss)
        self.log('train_acc', self.train_acc)
        self.log('train_auc_roc', self.train_auc_roc)
        self.log('train_average_precision', self.train_average_precision)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        scores, loss = self.scores_loss(x, y)
        self.val_acc(scores, y)
        # Need to detach scores when passing to torchmetrics.AUROC or torchmetrics.AveragePrecision objects
        # to avoid deepcopy error.
        self.val_auc_roc(scores.detach(), y)
        self.val_average_precision(scores.detach(), y)
        return {'val_loss': loss}

    def validation_epoch_end(self,
                             validation_step_outputs: Sequence[Dict[str, torch.Tensor]]) -> None:
        self._curr_val_loss = torch.tensor(
            [batch['val_loss'] for batch in validation_step_outputs]).mean()
        self._curr_val_acc = self.val_acc.compute()
        self.log('val_loss', self._curr_val_loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auc_roc', self.val_auc_roc)
        self.log('val_average_precision', self.val_average_precision)

        # Update and log the best val acc and loss seen so far.
        # booleans used if save_best_model.
        is_best_val_loss = self._curr_val_loss < self.best_val_loss
        is_best_val_acc = self._curr_val_acc > self.best_val_acc
        self.best_val_loss = min(self.best_val_loss, self._curr_val_loss)
        self.best_val_acc = max(self.best_val_acc, self._curr_val_acc)
        self.log('best_val_acc', self.best_val_acc, prog_bar=True)
        self.log('best_val_loss', self.best_val_loss, prog_bar=True)

        # Save best models, if desired/implementd.
        if is_best_val_acc and self.save_models_to_wandb:
            self.save_best_model('val_acc')
        if is_best_val_loss and self.save_models_to_wandb:
            self.save_best_model('val_loss')

    def save_best_model(self,
                        metric: str) -> None:
        """Overwrite.

        Save both to onnx format and the state_dict to wandb based on metric,
        which is either 'val_loss' or 'val_acc'.
        """
        raise NotImplementedError(
            "Must overwrite the save_best_model method.")

    def test_step(self):
        raise NotImplementedError('Should not be using test dataset yet.')

    def test_epoch_end(self):
        raise NotImplementedError('Should not be using test dataset yet.')


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
    `seq_len`: int
        Length of text (counted by number of characters.)
    `tokens`: str or pd.DataFrame
        DataFrame object or path to character feather file, determines the size
        of the one-hot vectors.
    `fc_dims`: Sequence[int], optional, default = None
        Tuple of hidden dimension sizes. Default implements [64, 32].
    `lr`: float, optional, default = 1e-3
        Learning rate for Adam optimizer.
    `nonlinearity`: bool, optional, default = True
        Boolean for whether to use insert ReLUs between hidden layers. A flag to
        test linear models.
    `save_models_to_wandb`: bool, optional, default = False
        Toggles saving the best models according to validation accuracy or
        loss to wandb.
    `logging_kwargs`: None or dict, optional
        Optional kwargs which don't affect performance, but which will be
        tracked by loggers such as wandb. Useful for tracking batch size, e.g.
    """

    def __init__(self,
                 seq_len: int,
                 tokens: Union[str, pd.DataFrame],
                 fc_dims: Sequence[int] = None,
                 lr: Optional[float] = 1e-3,
                 nonlinearity: Optional[bool] = True,
                 save_models_to_wandb: Optional[bool] = False,
                 **logging_kwargs: Dict[str, Union[float, int, str]]) -> None:
        super().__init__(seq_len, save_models_to_wandb)
        # Logging hyperparameters to the hparams attr of the class (pl feature)
        # saves all args of __init__ to self.hparam.  So, can get the lr via
        # self.hparams['lr'], for instance.
        self.save_hyperparameters(ignore='tokens')

        # Get input_size from tokens file.
        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = tokens
        self.hparams['input_size'] = len(self.tokens_df)

        # batch_size = 1 dummy input needed for onnx saving.
        self.dummy_input = torch.randn(1, seq_len, self.hparams['input_size'])

        if not fc_dims:
            self.fc_dims = []
        else:
            self.fc_dims = fc_dims
        # Zip in and out dimensions together
        flattened_size = seq_len * self.hparams['input_size']
        in_out_dims = zip([flattened_size] + self.fc_dims,
                          self.fc_dims + [1])
        self.fc_layers = [nn.Linear(i, o) for i, o in in_out_dims]
        # Insert ReLUs between all FC layers, if nonlinearity.
        if nonlinearity:
            self.layers = [None for _ in range(2 * len(self.fc_layers) - 1)]
            self.layers[::2] = self.fc_layers
            self.layers[1::2] = [nn.ReLU()
                                 for _ in range(len(self.fc_layers) - 1)]
            self.layers = nn.ModuleList(self.layers)
        else:
            self.layers = nn.ModuleList(self.fc_layers)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        # Flatten and pass through layers.
        seq_len, input_size = self.hparams['seq_len'], self.hparams['input_size']
        output = input.view(-1, seq_len * input_size)
        output = (output - output.mean()) / output.std()
        for idx, layer in enumerate(self.layers):
            output = layer(output)
        return output.view(-1)

    def scores_loss(self,
                    inputs: torch.Tensor,
                    targets: torch.Tensor) -> Tuple[torch.Tensor]:
        scores = self(inputs)
        loss = F.binary_cross_entropy_with_logits(scores, targets.float())
        return scores, loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])

    def save_best_model(self,
                        metric: str) -> None:
        # Assumes that a dummy_input attr was provided to __init__
        assert hasattr(
            self, 'dummy_input'), "dummy_input attr, a tensor of the appropriate architecture input shape (batch_size = 1), must be set to use save_best_model method."

        model_name = 'model_best_' + metric
        model_name_onnx = model_name + '.onnx'
        model_json_name = model_name + '_hparam_dict.json'
        model_state_dict_name = model_name + '.pt'

        print(f'Saved best {metric} at global step: {self.global_step}',
              f'Epoch: {self.current_epoch}',
              f'Validation accuracy: {self.val_acc.compute().item()}',
              f'Validation Loss: {self._curr_val_loss.item()}',
              sep='\n')

        # Dummy inputs all have batch_size 1.
        # We label these as dynamic_axes below.
        dummy_input = self.dummy_input.to(self.device)

        torch.onnx.export(self,
                          dummy_input,
                          model_name_onnx,
                          opset_version=12,
                          export_params=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}}
                          )

        wandb.save(model_name_onnx)
        # Also save the hyperparameters, accuracy, and loss.
        model_dict = {**{'val_loss': self._curr_val_loss.item(), 'val_acc': self._curr_val_acc.item()},
                      **self.hparams}
        model_dict_json = json.dumps(model_dict)
        with open(model_json_name, "w") as outfile:
            outfile.write(model_dict_json)
        wandb.save(model_json_name)
        torch.save(self.state_dict(), model_state_dict_name)
        wandb.save(model_state_dict_name)
