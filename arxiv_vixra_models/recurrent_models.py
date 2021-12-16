import json
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
from typing import Dict, List, Tuple, Sequence, Optional, Union
from typing_extensions import Literal
import wandb


class LitRecurrentLoggingBase(pl.LightningModule):
    """
    Custom PyTorch Lightning base class for recurrent architectures with
    logging. Defining common methods for train, validation, and test steps.

    Description
    ----------
    Written such that one-hot and embedding models need only subclass 
    LitRecurrentLoggingBase with __init__ method overwritten.  Expects
    fully-connected layers after recurrent structure to be in a ModuleList
    instance and named fc_layers.

    In order to use save_best_model, the inheriting class must create a
    dummy_input attr, a torch.tensor in the appropriate shape of the input
    (with batch_size = 1) for the architecture.

    Args
    ----------
    `seq_len`: int
        Length of sequences passed to the recurrent architecture.
    `save_models_to_wandb`: bool, optional, default = False
        Toggles saving the best models according to validation accuracy or
        loss to wandb.
    `lr`: float, optional, default = 1e-3
        Learning rate for Adam optimizer.
    `l2_reg`: float, optional, default = 0.
        Parameter controlling strength of l2 regularization on fully-connected
        layer weights.
    `hidden_strategy`: in ('last', 'max', 'mean'), optional, default = 'last'
        Determines which information from the hidden states is propagated to
        the fully-connected classification layers: the state from the last step,
        the max element across all steps, or the mean of all steps.
    """

    def __init__(self,
                 seq_len: int,
                 save_models_to_wandb: Optional[bool] = False,
                 lr: Optional[float] = 1e-3,
                 l2_reg: Optional[float] = 0.,
                 hidden_strategy: Literal['last', 'max', 'mean'] = 'last') -> None:
        super().__init__()
        self.seq_len = seq_len
        self.save_models_to_wandb = save_models_to_wandb
        self.lr = lr
        self.l2_reg = l2_reg
        self.hidden_strategy = hidden_strategy

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

    def configure_optimizers(self):
        """Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self,
                input: torch.Tensor,
                hiddens: Optional[Union[torch.Tensor, None]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output, hiddens = self.rnn(input, hiddens)
        assert self.hidden_strategy in (
            'last', 'max', 'mean'), 'hidden_strategy must be one of last, max, or mean.'
        if self.hidden_strategy == 'last':
            output = output[:, -1]
        elif self.hidden_strategy == 'max':
            output = output.max(dim=1)
        else:
            output = output.mean(dim=1)
        for layer in self.fc_layers:
            output = layer(output)
        # Return the output as a (batch_size,) vector of raw scores.
        return output.view(-1), hiddens

    def scores_loss_hiddens(self,
                            inputs: torch.Tensor,
                            targets: torch.Tensor,
                            hiddens: Optional[Union[torch.Tensor, None]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, hiddens = self(inputs, hiddens)
        loss = F.binary_cross_entropy_with_logits(scores, targets.float())
        loss += self.l2_reg * self.class_layer ** 2
        return scores, loss, hiddens

    def configure_optimizers(self):
        """Returns an instance of a torch.optim class, such as Adam.
        """
        raise NotImplementedError('Must overwite configure_optimizers method!')

    def inference(self,
                  input: torch.Tensor,
                  hiddens: Optional[Union[torch.Tensor, None]] = None) -> torch.Tensor:
        """Inference mode with eval set and no_grad context included. Only
        returns probs.
        """
        self.eval()
        with torch.no_grad():
            scores, hiddens = self(input, hiddens)
        return scores.sigmoid()

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int,
                      hiddens: Optional[Union[torch.Tensor, None]] = None) -> Dict[str, torch.Tensor]:
        """Currently calls self.scores_loss_hiddens with hiddens=None which
        re-initializes hiddens to a new zeros tensor at every training step.
        This may be undesireable, depending on application.
        """
        inputs, targets = batch
        iddens = self.scores_loss_hiddens(inputs, targets, hiddens=None)
        self.train_acc(scores, targets)
        # Need to detach scores when passing to torchmetrics.AUROC or torchmetrics.AveragePrecision objects
        # to avoid deepcopy error.
        self.train_auc_roc(scores.detach(), targets)
        self.train_average_precision(scores.detach(), targets)
        return {'loss': loss, 'hiddens': hiddens}

    def training_epoch_end(self,
                           training_step_outputs: Dict[str, torch.Tensor]) -> None:
        """If tbptt_split_batch is used, training_step_outputs will contain lists of lists of dicts, 
        rather than single lists of dicts, so a flattening helper function must be used before
        computing the mean loss.
        """
        mean_loss = torch.tensor(
            [batch['loss'] for batch in self._flatten_2d_list(training_step_outputs)]).mean()
        self.log('train_loss', mean_loss)
        self.log('train_acc', self.train_acc)
        self.log('train_auc_roc', self.train_auc_roc)
        self.log('train_average_precision', self.train_average_precision)

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int) -> Dict[str, torch.Tensor]:
        inputs, targets = batch
        scores, loss, _ = self.scores_loss_hiddens(inputs, targets)
        self.val_acc(scores, targets)
        # Need to detach scores when passing to torchmetrics.AUROC or torchmetrics.AveragePrecision objects
        # to avoid deepcopy error.
        self.val_auc_roc(scores.detach(), targets)
        self.val_average_precision(scores.detach(), targets)
        return {'val_loss': loss}

    def validation_epoch_end(self,
                             validation_step_outputs: Dict[str, torch.Tensor]) -> None:
        """
        If tbptt_split_batch is used, validation_step_outputs will contain lists of lists of dicts, 
        rather than single lists of dicts, so a flattening helper function must be used before
        computing the mean loss.
        """
        self._curr_val_loss = torch.tensor(
            [batch['val_loss'] for batch in self._flatten_2d_list(validation_step_outputs)]).mean()
        self._curr_val_acc = self.val_acc.compute()
        self.log('val_loss', self._curr_val_loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_auc_roc', self.val_auc_roc)
        self.log('val_average_precision', self.val_average_precision)

        # Update and log the best val acc and loss seen so far.
        # booleans used if save_best_model
        is_best_val_loss = self._curr_val_loss < self.best_val_loss
        is_best_val_acc = self._curr_val_acc > self.best_val_acc
        self.best_val_loss = min(self.best_val_loss, self._curr_val_loss)
        self.best_val_acc = max(self.best_val_acc, self._curr_val_acc)
        self.log('best_val_acc', self.best_val_acc, prog_bar=True)
        self.log('best_val_loss', self.best_val_loss, prog_bar=True)

        if is_best_val_acc and self.save_models_to_wandb:
            self.save_best_model('val_acc')
        if is_best_val_loss and self.save_models_to_wandb:
            self.save_best_model('val_loss')

    def save_best_model(self,
                        metric: str) -> None:
        """Save best models by metric both to onnx and .pt (state_dict) format.
        """
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

        # For recurrent networks one get a UswerWarning of the form
        # "UserWarning: Exporting a model to ONNX with a batch_size..."
        # This can apparently be ignored.
        # https://github.com/pytorch/pytorch/issues/45653

        # Dummy inputs all have batch_size 1.
        # We label these as dynamic_axes below.
        bidirectional_factor = 1 + self.hparams['bidirectional']
        dummy_input = self.dummy_input.to(self.device)
        dummy_hiddens = torch.randn(
            (bidirectional_factor * self.hparams['num_layers'], 1, self.hparams['hidden_size']), device=self.device)
        dummy_cell = torch.randn(
            (bidirectional_factor * self.hparams['num_layers'], 1, self.hparams['hidden_size']), device=self.device)

        if self.hparams['rnn_type'] == 'LSTM':
            dummy_input_args = (dummy_input, (dummy_hiddens, dummy_cell))
        else:
            dummy_input_args = (dummy_input, dummy_hiddens)

        torch.onnx.export(self,
                          dummy_input_args,
                          model_name_onnx,
                          opset_version=12,
                          export_params=True,
                          input_names=['input', 'hiddens_in'],
                          output_names=['output', 'hiddens_out'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'hiddens_in': {1: 'batch_size'},
                                        'output': {0: 'batch_size'},
                                        'hiddens_out': {1: 'batch_size'}}
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

    def test_step(self):
        raise NotImplementedError('Should not be using test dataset yet.')

    def test_epoch_end(self):
        raise NotImplementedError('Should not be using test dataset yet.')

    def tbptt_split_batch(self, 
                          batch: Tuple[torch.Tensor, torch.Tensor],
                          split_size: Union[int, float]) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Passing a float to split_size chooses the split via a fraction of the
        sequence length. If used, by setting truncated_bptt_steps to not None,
        the shape of the x_step_outputs that x_step passes to x_epoch_end is
        altered from a list of dictionaries (of len determined by batch size)
        to a list (of len determined by batch size) of lists (determined by the
        number of tbptt splits) of dictionaries.
        """
        if type(split_size) == float:
            assert 0 <= split_size <= 1, "split_size must be between 0 and 1"
            split_size = int(split_size * self.seq_len)

        one_hot_text, source = batch
        time_dims = [x.shape[0] for x in one_hot_text]
        assert len(time_dims) >= 1, "Unable to determine batch time dimension"
        assert all(
            x == time_dims[0] for x in time_dims), "Batch time dimension length is ambiguous"
        # Overwrote the splitting procedure here.
        splits = []
        for t in range(0, time_dims[0], split_size):
            split_text = one_hot_text[:, t:t + split_size]
            splits.append((split_text, source))
        return splits

    def _flatten_2d_list(self, l):
        """Helper function for recursively flattening 2D lists. General recursive approach avoided,
        as it can hit recursion limit."""
        flat_list = []
        for item in l:
            if isinstance(item, list):
                for sub_item in item:
                    flat_list.append(sub_item)
            else:
                flat_list.append(item)
        return flat_list


class LitOneHotCharRecurrent(LitRecurrentLoggingBase):
    """Recurrent architectures for one-hot encoded data built atop
    LitRecurrentLoggingBase class.

    Description
    ----------

    Must specify `rnn_type` which can be any of `RNN`, `LSTM` or `GRU`.

    Trained on text which had .strip, unidecode, and .lower
    applied to it.

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
    `num_layers`: int, optional, default = 1
        Number of rnn layers; standard torch rnn arg.
    `hidden_size`: int, optional, default = 128
        Hidden dimension size; standard torch rnn arg.
    `hidden_strategy`: in ('last', 'max', 'mean'), optional, default = 'last'
        Determines which information from the hidden states is propagated to
        the fully-connected classification layers: the state from the last step,
        the max element across all steps, or the mean of all steps.
    `bidirectional`: bool, optional, default = False
        Flag for whether the architecture is bidirectional
    `lr`: float, optional, default = 1e-3
        Learning rate for Adam optimizer.
    `l2_reg`: float, optional, default = 0.
        Parameter controlling strength of l2 regularization on fully-connected
        layer weights.
    `fc_dropout`: None or float, optional, default = None
        Set dropout probability before final fully-connected layer, if any.
    `recurrent_dropout`: None or float, optional, default = None
        Set dropout probability between recurrent layers, if any.
    `truncated_bptt_steps`: None or int or float, optional, default = None
        Implements truncated backpropagation through time, if provided.
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
                 rnn_type: Literal['RNN', 'LSTM', 'GRU'],
                 num_layers: Optional[int] = 1,
                 hidden_size: Optional[int] = 128,
                 hidden_strategy: Literal['last', 'max', 'mean'] = 'last',
                 bidirectional: Optional[bool] = False,
                 lr: Optional[float] = 1e-3,
                 l2_reg: Optional[float] = 0.,
                 fc_dims: Optional[Union[List[int], None]] = None,
                 fc_dropout: Optional[Union[float, None]] = None,
                 recurrent_dropout: Optional[Union[float, None]] = None,
                 truncated_bptt_steps: Optional[Union[int, None]] = None,
                 save_models_to_wandb: Optional[bool] = False,
                 **logging_kwargs: Dict[str, Union[float, int, str]]) -> None:
        super().__init__(seq_len, save_models_to_wandb, lr, l2_reg, hidden_strategy)

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

        # The fully connected layer size depends on the bidirectional flag.
        self._fc_input_size = (2 if bidirectional else 1) * hidden_size
        # Need to set truncated_bptt_steps as an attr to implement tbptt.
        if truncated_bptt_steps is not None:
            self.truncated_bptt_steps = truncated_bptt_steps
        self.save_models_to_wandb = save_models_to_wandb

        # Set up the appropriate reccurent layer.
        assert rnn_type in (
            'RNN', 'LSTM', 'GRU'), "rnn_type must be one of RNN, LSTM, or GRU."
        rnn_dict = {'RNN': nn.RNN,
                    'LSTM': nn.LSTM,
                    'GRU': nn.GRU}
        self.rnn = rnn_dict[rnn_type]
        self._rnn_dict = {'input_size': self.hparams['input_size'],
                          'hidden_size': hidden_size,
                          'num_layers': num_layers,
                          'batch_first': True,
                          'bidirectional': bidirectional
                          }
        if recurrent_dropout is not None:
            self._rnn_dict['dropout'] = None
        self.rnn = self.rnn(**self._rnn_dict)

        in_out_dims = zip([self._fc_input_size] + self.fc_dims,
                          self.fc_dims + [1])
        self.linears = [nn.Linear(i, o) for i, o in in_out_dims]
        # Insert ReLUs and Dropout (if fc_dropout) between all FC layers.
        if self.hparams['fc_dropout']:
            self.fc_layers = [None for _ in range(3 * len(self.linears) - 2)]
            self.fc_layers[::3] = self.linears
            self.fc_layers[1::3] = [nn.ReLU()
                                    for _ in range(len(self.linears) - 1)]
            self.fc_layers[2::3] = [nn.Dropout(
                self.hparams['fc_dropout']) for _ in range(len(self.linears) - 1)]
            self.fc_layers = nn.ModuleList(self.fc_layers)
        else:
            self.fc_layers = [None for _ in range(2 * len(self.linears) - 1)]
            self.fc_layers[::2] = self.linears
            self.fc_layers[1::2] = [nn.ReLU()
                                    for _ in range(len(self.linears) - 1)]
            self.fc_layers = nn.ModuleList(self.fc_layers)


class LitEmbeddingRecurrent(LitRecurrentLoggingBase):
    """Recurrent architectures for embedded text built atop
    LitRecurrentLoggingBase class.

    Description
    ----------

    Must specify `rnn_type` which can be any of `RNN`, `LSTM` or `GRU`.

    Trained on text which had .strip, unidecode, and .lower
    applied to it.

    Test-data methods are currently not implemented to prevent early peeking.

    Args
    ----------
    `seq_len`: int
        Sequence length; standard torch rnn arg.
    `tokens`: str or pd.DataFrame
        DataFrame object or path to vocabulary feather file. Used to determine
        num_embeddings arg of nn.Embedding.
    `rnn_type`: in ('RNN', 'LSTM', 'GRU')
        RNN architecture.
    `embedding_dim`: int, optional, default = 256
        Size of embedding dimension; standard torch Embedding arg.
    `num_layers`: int, optional, default = 1
        Number of rnn layers; standard torch rnn arg.
    `hidden_size`: int, optional, default = 128
        Hidden dimension size; standard torch rnn arg.
    `hidden_strategy`: in ('last', 'max', 'mean'), optional, default = 'last'
        Determines which information from the hidden states is propagated to
        the fully-connected classification layers: the state from the last step,
        the max element across all steps, or the mean of all steps.
    `bidirectional`: bool, optional, default = False
        Flag for whether the architecture is bidirectional
    `lr`: float, optional, default = 1e-3
        Learning rate for Adam optimizer.
    `l2_reg`: float, optional, default = 0.
        Parameter controlling strength of l2 regularization on final
        fully-connected layer weights.
    `fc_dropout`: None or float, optional, default = None
        Set dropout probability before final fully-connected layer, if any.
    `recurrent_dropout`: None or float, optional, default = None
        Set dropout probability between recurrent layers, if any.
    `truncated_bptt_steps`: None or int or float, optional, default = None
        Implements truncated backpropagation through time, if provided.
    `save_models_to_wandb`: bool, optional, default = False
        Toggles saving the best models according to validation accuracy and
        loss to wandb.
    `logging_kwargs`: None or dict, optional
        Optional kwargs which don't affect performance, but which will be
        tracked by loggers such as wandb. Useful for tracking batch size, e.g.
     """

    def __init__(self,
                 seq_len: int,
                 tokens: Union[str, pd.DataFrame],
                 rnn_type: Literal['RNN', 'LSTM', 'GRU'],
                 embedding_dim: Optional[int] = 256,
                 num_layers: Optional[int] = 1,
                 hidden_size: Optional[int] = 128,
                 hidden_strategy: Literal['last', 'max', 'mean'] = 'last',
                 bidirectional: Optional[bool] = False,
                 lr: Optional[float] = 1e-3,
                 l2_reg: Optional[float] = 0.,
                 fc_dims: Optional[Union[List[int], None]] = None,
                 fc_dropout: Optional[Union[float, None]] = None,
                 recurrent_dropout: Optional[Union[float, None]] = None,
                 truncated_bptt_steps: Optional[Union[int, None]] = None,
                 save_models_to_wandb: Optional[bool] = False,
                 **logging_kwargs: Dict[str, Union[float, int, str]]) -> None:
        # Pass seq_len and logging_kwargs
        # to LitRecurrentLoggingBase parent class.
        super().__init__(seq_len, save_models_to_wandb, lr, l2_reg, hidden_strategy)

        # Logging hyperparameters to the hparams attr of the class (pl feature)
        # saves all args of __init__ to self.hparam.  So, can get the lr via
        # self.hparams['lr'], for instance.
        self.save_hyperparameters(ignore='tokens')

        # Get num_embeddings from tokens
        if isinstance(tokens, str):
            self.tokens_df = pd.read_feather(tokens)
        else:
            self.tokens_df = tokens
        # + 2 below for padding and unknown words.
        self.hparams['num_embeddings'] = len(self.tokens_df) + 2

        # batch_size = 1 dummy input needed for onnx saving.
        self.dummy_input = torch.randint(
            self.hparams['num_embeddings'], (1, seq_len))

        # The fully connected layer size depends on the bidirectional flag.
        self._fc_input_size = (2 if bidirectional else 1) * hidden_size
        # Need to set truncated_bptt_steps as an attr to implement tbptt.
        if truncated_bptt_steps is not None:
            self.truncated_bptt_steps = truncated_bptt_steps
        self.save_models_to_wandb = save_models_to_wandb

        # Embedding layer with 0 as padding idx.
        self.embedding = nn.Embedding(num_embeddings=self.hparams['num_embeddings'],
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)
        # Set up the appropriate reccurent layer.
        assert rnn_type in (
            'RNN', 'LSTM', 'GRU'), "rnn_type must be one of RNN, LSTM, or GRU."
        rnn_dict = {'RNN': nn.RNN,
                    'LSTM': nn.LSTM,
                    'GRU': nn.GRU}
        self.rnn = rnn_dict[rnn_type]
        self._rnn_dict = {'input_size': embedding_dim,
                          'hidden_size': hidden_size,
                          'num_layers': num_layers,
                          'batch_first': True,
                          'bidirectional': bidirectional
                          }
        if recurrent_dropout is not None:
            self._rnn_dict['dropout'] = recurrent_dropout
        self.rnn = self.rnn(**self._rnn_dict)

        in_out_dims = zip([self._fc_input_size] + self.fc_dims,
                          self.fc_dims + [1])
        self.linears = [nn.Linear(i, o) for i, o in in_out_dims]
        # Insert ReLUs and Dropout (if fc_dropout) between all FC layers.
        if self.hparams['fc_dropout']:
            self.fc_layers = [None for _ in range(3 * len(self.linears) - 2)]
            self.fc_layers[::3] = self.linears
            self.fc_layers[1::3] = [nn.ReLU()
                                    for _ in range(len(self.linears) - 1)]
            self.fc_layers[2::3] = [nn.Dropout(
                self.hparams['fc_dropout']) for _ in range(len(self.linears) - 1)]
            self.fc_layers = nn.ModuleList(self.fc_layers)
        else:
            self.fc_layers = [None for _ in range(2 * len(self.linears) - 1)]
            self.fc_layers[::2] = self.linears
            self.fc_layers[1::2] = [nn.ReLU()
                                    for _ in range(len(self.linears) - 1)]
            self.fc_layers = nn.ModuleList(self.fc_layers)
