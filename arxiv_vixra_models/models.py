from . import *

class LitRecurrentLoggingBase(LightningModule):
    """
    PyTorch Lightning base class defining common methods for train, validation,
    and test steps for recurrent architectures with logging.

    Description
    ----------

    Must overwrite the following methods to use:
    - forward
    - preds_losses_hiddens
    - configure optimizers
    - save_best_model
    - tbptt_split_batch

    In order to use save_best_model, the inheriting class must define a dummy_input
    attr, a torch.tensor in the appropriate shape of the input (with batch_size = 1)
    for the architecture.

    Args
    ----------
    `seq_len`: int
        Length of sequences passed to the recurrent architecture.
    """

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        
        # Attrs to store the best val_acc and val_loss seen at the end of any
        # validation epoch
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.

        # torchmetrics methods for computing various statistics.
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        
        self.train_auc_roc = AUROC()
        self.val_auc_roc = AUROC()
        self.test_auc_roc = AUROC()
        
        self.train_average_precision = AveragePrecision()
        self.val_average_precision = AveragePrecision()
        self.test_average_precision = AveragePrecision()
        
    def forward(self,
                input: Tensor,
                hiddens: Optional[Union[Tensor, None]] = None) -> Tensor:
        """
        Overwrite. Expected to return (outputs, hiddens) tuple.
        """
        raise NotImplementedError("Must overwrite the forward method.")

    def preds_losses_hiddens(self, 
                             inputs: Tensor,
                             targets: Tensor,
                             hiddens: Optional[Union[Tensor, None]] = None) -> Tuple[Tensor]:
        """
        Overwrite. Expected to return (probs, loss, hiddens) tuple.
        """
        raise NotImplementedError(
            "Must overwrite the preds_losses_hiddens method.")

    def configure_optimizers(self):
        """
        Overwrite.
        """
        raise NotImplementedError(
            "Must overwrite the configure_optimizers method.")

    def training_step(self, batch: Tensor,
                      batch_idx: int,
                      hiddens: Optional[Union[Tensor, None]] = None) -> Tuple[Tensor, Tensor]:
        x, y = batch
        # By calling self.preds_losses_hiddens with hiddens=None, we are
        # re-initializing hiddens to a new zeros tensor. 
        preds, loss, hiddens = self.preds_losses_hiddens(x, y, hiddens=None)
        self.train_acc(preds, y)
        # Need to detach preds when passing to AUROC or AveragePrecision objects
        # to avoid deepcopy error.
        self.train_auc_roc(preds.detach(), y)
        self.train_average_precision(preds.detach(), y)
        return {'loss': loss, 'hiddens': hiddens}

    def training_epoch_end(self, training_step_outputs: Tensor) -> None:
        """
        If tbptt_split_batch is used, training_step_outputs will contain lists of lists of dicts, 
        rather than single lists of dicts, so a flattening helper function must be used before
        computing the mean loss.
        """
        mean_loss = torch.tensor(
            [batch['loss'] for batch in self._flatten_2d_list(training_step_outputs)]).mean()
        self.log('train_loss', mean_loss)
        self.log('train_acc', self.train_acc)
        self.log('train_auc_roc', self.train_auc_roc)
        self.log('train_average_precision', self.train_average_precision)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor]:
        x, y = batch
        preds, loss, _ = self.preds_losses_hiddens(x, y)
        self.val_acc(preds, y)
        # Need to detach preds when passing to AUROC or AveragePrecision objects
        # to avoid deepcopy error.
        self.val_auc_roc(preds.detach(), y)
        self.val_average_precision(preds.detach(), y)
        return {'val_loss': loss}

    def validation_epoch_end(self, validation_step_outputs: Tensor) -> None:
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
        # booleans references if save_best_model
        is_best_val_loss = self._curr_val_loss < self.best_val_loss
        is_best_val_acc = self._curr_val_acc > self.best_val_acc
        self.best_val_loss = min(self.best_val_loss, self._curr_val_loss)
        self.best_val_acc = max(self.best_val_acc, self._curr_val_acc)
        self.log('best_val_acc', self.best_val_acc, prog_bar=True)
        self.log('best_val_loss', self.best_val_loss, prog_bar=True)

        if is_best_val_acc:
            self.save_best_model('val_acc')
        if is_best_val_loss:
            self.save_best_model('val_loss')

    def save_best_model(self, metric: str) -> None:
        """
        Save both to onnx format and the state_dict to wandb based on metric,
        which is either val_loss or val_acc.
        """
        # 
        if self.save_models_to_wandb:
            # Assumes that a dummy_input attr was provided to __init__
            assert hasattr(self, 'dummy_input'), "dummy_input attr, a tensor of the appropriate architecture input shape (batch_size = 1), must be set to use save_best_model method."

            model_name = 'model_best_' + metric + '.onnx'
            model_name_onnx = model_name + '.onnx'
            model_json_name = model_name + '_hparam_dict.json'
            model_state_dict_name = model_name + '_state_dict.pt'

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
                                            'hiddens_in': {1: 'batch_size'}}
                              )
            # wandb.save knows to save the just-written file, given the filename.
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

    def tbptt_split_batch(self, batch: Tuple[Tensor], split_size: Union[int, float]) -> Tuple[Tensor]:
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
    """
    Recurrent architectures for one-hot encoded data built atop
    LitRecurrentLoggingBase class.

    Description
    ----------

    Must specify `rnn_type` which can be any of `RNN`, `LSTM` or `GRU`.

    Trained on text which had .strip(), unidecode(), and .lower()
    applied to it.

    Test-data methods are currently not implemented to prevent early peeking.

    Args
    ----------
    `seq_len`: int
        Sequence length; standard torch rnn arg.
    `chars`: str or pd.DataFrame
        DataFrame object or path to character feather file. Used to determine
        input_size arg of recurrent layer.
    `rnn_type`: str, in ('RNN', 'LSTM', 'GRU')
        RNN architecture.
    `num_layers`: int, optional, default = 1
        Number of rnn layers; standard torch rnn arg.
    `hidden_size`: int, optional, default = 128
        Hidden dimension size; standard torch rnn arg.
    `bidirectional`: bool, optional, default = False
        Flag for whether the architecture is bidirectional
    `lr`: float, optional, default = 1e-4
        Learning rate for Adam optimizer.
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
                 chars: Union[str, DF],
                 rnn_type: Literal['RNN', 'LSTM', 'GRU'],
                 num_layers: Optional[int] = 1,
                 hidden_size: Optional[int] = 128,
                 bidirectional: Optional[bool] = False,
                 lr: Optional[float] = 1e-3,
                 fc_dropout: Optional[Union[float, None]] = None,
                 recurrent_dropout: Optional[Union[float, None]] = None,
                 truncated_bptt_steps: Optional[Union[int, None]] = None,
                 save_models_to_wandb: Optional[bool] = False,
                 **logging_kwargs: Dict[str, Union[float, int, str]]) -> None:
        # Pass seq_len and logging_kwargs
        # to LitRecurrentLoggingBase parent class.
        super().__init__(seq_len)

        # Logging hyperparameters to the hparams attr of the class (pl feature)
        # saves all args of __init__ to self.hparam.  So, can get the lr via
        # self.hparams['lr'], for instance.
        self.save_hyperparameters(ignore='chars')

        # Get input_size from chars file.
        if isinstance(chars, str):
            self.chars_df = pd.read_feather(chars)
        else:
            self.chars_df = chars
        self.hparams['input_size'] = len(self.chars_df)

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
            self._rnn_dict['dropout'] = recurrent_dropout
        self.rnn = self.rnn(**self._rnn_dict)
        if fc_dropout is not None:
            self.fc_dropout = nn.Dropout(fc_dropout)
        self.class_layer = nn.Linear(self._fc_input_size, 1)
        
    def forward(self,
                input: Tensor,
                hiddens: Optional[Union[Tensor, None]] = None) -> Tensor:
        # Center and normalize data according to one-hot dimension (input_size).
        mean = 1 / self.hparams['input_size']
        var = mean - mean ** 2 # <x^2> = <x> for one-hot data.
        std = torch.sqrt(var)
        z  = (input - mean) / std
        x, hiddens = self.rnn(z, hiddens)
        # Isolate the rnn output from the final time step
        x = x[:, -1]
        # Use dropout, if instantiated.
        if hasattr(self, 'fc_dropout'):
            x = self.fc_dropout(x)
        # Pass the raw scores and use binary_cross_entropy_with_logits for loss.
        x = self.class_layer(x)
        # Return the output as a (B, ) vector
        return x.view(-1), hiddens

    def preds_losses_hiddens(self, 
                             inputs: Tensor,
                             targets: Tensor,
                             hiddens: Optional[Union[Tensor, None]] = None) -> Tuple[Tensor]:
        preds, hiddens = self(inputs, hiddens)
        loss = F.binary_cross_entropy_with_logits(preds, targets.float())
        return preds, loss, hiddens

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])

class LitEmbeddingRecurrent(LitRecurrentLoggingBase):
    """
    Recurrent architectures for embedded text built atop
    LitRecurrentLoggingBase class.

    Description
    ----------

    Must specify `rnn_type` which can be any of `RNN`, `LSTM` or `GRU`.

    Trained on text which had .strip(), unidecode(), and .lower()
    applied to it.

    Test-data methods are currently not implemented to prevent early peeking.

    Args
    ----------
    `seq_len`: int
        Sequence length; standard torch rnn arg.
    `vocab`: str or pd.DataFrame
        DataFrame object or path to vocabulary feather file. Used to determine
        num_embeddings arg of nn.Embedding.
    `rnn_type`: str, in ('RNN', 'LSTM', 'GRU')
        RNN architecture.
    `embedding_dim`: int, optional, default = 256
        Size of embedding dimension; standard torch Embedding arg.
    `num_layers`: int, optional, default = 1
        Number of rnn layers; standard torch rnn arg.
    `hidden_size`: int, optional, default = 128
        Hidden dimension size; standard torch rnn arg.
    `bidirectional`: bool, optional, default = False
        Flag for whether the architecture is bidirectional
    `lr`: float, optional, default = 1e-4
        Learning rate for Adam optimizer.
    `input_dropout`: None or float, optional, default = None
        Set dropout probability after the embedding layer, if any.
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
                 vocab: Union[str, DF],
                 rnn_type: Literal['RNN', 'LSTM', 'GRU'],
                 embedding_dim: Optional[int] = 256,
                 num_layers: Optional[int] = 1,
                 hidden_size: Optional[int] = 128,
                 bidirectional: Optional[bool] = False,
                 lr: Optional[float] = 1e-3,
                 input_dropout: Optional[Union[float, None]] = None,
                 fc_dropout: Optional[Union[float, None]] = None,
                 recurrent_dropout: Optional[Union[float, None]] = None,
                 truncated_bptt_steps: Optional[Union[int, None]] = None,
                 save_models_to_wandb: Optional[bool] = False,
                 **logging_kwargs: Dict[str, Union[float, int, str]]) -> None:
        # Pass seq_len and logging_kwargs
        # to LitRecurrentLoggingBase parent class.
        super().__init__(seq_len)
        
        # Logging hyperparameters to the hparams attr of the class (pl feature)
        # saves all args of __init__ to self.hparam.  So, can get the lr via
        # self.hparams['lr'], for instance.
        self.save_hyperparameters(ignore='vocab')

        # Get num_embeddings from vocab
        if isinstance(vocab, str):
            self.vocab_df = pd.read_feather(vocab)
        else:
            self.vocab_df = vocab
        # + 2 below for padding and unknown words.
        self.hparams['num_embeddings'] = len(self.vocab_df) + 2

        # batch_size = 1 dummy input needed for onnx saving.
        self.dummy_input = torch.randint(self.hparams['num_embeddings'], (1, seq_len))

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
        # Dropout setup.
        if input_dropout is not None:
            self.input_dropout = nn.Dropout(input_dropout)
        if fc_dropout is not None:
            self.fc_dropout = nn.Dropout(fc_dropout)
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
        

        self.class_layer = nn.Linear(self._fc_input_size, 1)

    def forward(self,
                input: Tensor,
                hiddens: Optional[Union[Tensor, None]] = None) -> Tensor:

        x = self.embedding(input)
        if hasattr(self, 'input_dropout'):
            x = self.input_dropout(x)
        x, hiddens = self.rnn(x, hiddens)
        # Isolate the rnn output from the final time step
        x = x[:, -1]
        # Use dropout, if instantiated.
        if hasattr(self, 'fc_dropout'):
            x = self.fc_dropout(x)
        # Pass the raw scores and use binary_cross_entropy_with_logits for loss.
        x = self.class_layer(x)
        # Return the output as a (B, ) vector
        return x.view(-1), hiddens

    def preds_losses_hiddens(self, inputs: Tensor,
                             targets: Tensor,
                             hiddens: Optional[Union[Tensor, None]] = None) -> Tuple[Tensor, Tensor]:
        preds, hiddens = self(inputs, hiddens)
        loss = F.binary_cross_entropy_with_logits(preds, targets.float())
        return preds, loss, hiddens

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])