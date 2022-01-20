from copy import deepcopy
from time import perf_counter
from typing import Dict, Tuple, Union, Optional
from typing_extensions import Literal
import warnings

from pandas import DataFrame
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import wandb

from .datasets import GloVeDataset, CoMatrixDataset
from .embedding_utils import word_to_idx_dict_from_df

PAD_IDX = 0


class LitGloVe(pl.LightningModule):
    """PyTorch Lightning class for the GloVe algorithm.

    Parameters
    ----------
    co_matrix_sparse : Tensor
        Co-occurrence matrix, a sparse torch tensor.
    seq_len : int
        Sequence length; standard torch rnn arg.
    embedding_dim : int, default 512
        Size of embedding dimension; standard torch nn.Embedding arg.
    x_max : int, default 100,
        x_max loss function parameter, as defined in GloVe paper.
    alpha : float, default .75,
        alpha loss function parameter, as defined in GloVe paper.
    lr : float, default 5e-2
        Learning rate for Adam optimizer. Set to the GloVe default value.
    lr_scheduler : in ('cyclic', 'plateau', None), default None
        Toggles the use of a learning rate scheduler, either 'cyclic'
        (CyclicLR), 'plateau' (ReduceLROnPlateau), or None (constant lr).
    lr_scheduler_args : dict or None, default None
        Optional dictionary of arguments to be passed to the lr_scheduler.
        Overwrites every arg of the scheduler, apart from the optimizer.
    save_models_to_wandb : bool, default False
        Toggles saving the best models according to validation accuracy and
        loss to wandb.
    batch_size : int, default 128
        Dataloader batch_size arg.
    num_workers : bool, default True
        Dataloader num_workers arg.
    persistent_workers : bool, default True
        Dataloader persistent_workers arg.
    logging_kwargs : None or dict, optional
        Optional kwargs which don't affect performance, but which will be
        tracked by loggers such as wandb. Useful for tracking batch size, e.g.

    Methods
    ----------
    get_embedding_weights :
        Returns the mean of the two GloVe embedding layer weights.
    get_embedding_biases :
        Returns the mean of the two GloVe biases.
    get_embedding_vector :
        Returns the mean of the two Glove vectors corresponding to a given
        word index.
    get_unit_embedding_vector :
        Returns the mean of the two Glove vectors corresponding to a given
        word index whose length has been normalized to 1.
    get_embedding_cosine :
        Returns the cosine between the vectors corresponding to a pair of
        word indices.
    get_embedding_dot :
        Returns the dot-product between the vectors corresponding to a pair
        of word indices.
    pretrained_word_embedding : Tensor, optional
        Optionally provide a pretrained tensor for fine-tuning.
    pretrained_context_embedding : Tensor, optional
        Optionally provide a pretrained tensor for fine-tuning.
    pretrained_word_bias`: Tensor, optional
        Optionally provide a pretrained tensor for fine-tuning.
    pretrained_context_bias : Tensor, optional
        Optionally provide a pretrained tensor for fine-tuning.
    """

    def __init__(
        self,
        co_matrix_sparse: Tensor,
        embedding_dim: int = 512,
        x_max: int = 100,
        alpha: float = 0.75,
        lr: float = 5e-2,
        lr_scheduler: Literal["cyclic", "plateau", None] = None,
        lr_scheduler_args: Union[Dict[str, float], None] = None,
        lr_scheduler_interval: Literal["step", "epoch"] = "step",
        lr_scheduler_monitor: str = "train_batch_loss",
        save_models_to_wandb: bool = False,
        batch_size: int = 128,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        pretrained_word_embedding: Optional[Tensor] = None,
        pretrained_context_embedding: Optional[Tensor] = None,
        pretrained_word_bias: Optional[Tensor] = None,
        pretrained_context_bias: Optional[Tensor] = None,
        **logging_kwargs: Dict[str, Union[float, int, str]],
    ) -> None:
        super().__init__()
        # Save __init__ parameters to hparam dict attr.
        # Need to ignore co_matrix_sparse to avoid logging issues.
        self.save_hyperparameters(ignore="co_matrix_sparse")
        self.co_matrix_sparse = co_matrix_sparse
        # Key off of loss when saving models.
        self.best_loss = float("inf")

        # Get num_embeddings from co_matrix.
        self.hparams["num_embeddings"] = self.co_matrix_sparse.shape[-1]
        if pretrained_word_embedding is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=PAD_IDX
            )
        else:
            self.word_embedding = nn.Embedding(
                num_embeddings=self.hparams["num_embeddings"],
                embedding_dim=embedding_dim,
                padding_idx=PAD_IDX,
            )

        if pretrained_context_embedding is not None:
            self.context_embedding = nn.Embedding.from_pretrained(
                pretrained_context_embedding, freeze=False, padding_idx=PAD_IDX
            )
        else:
            self.context_embedding = nn.Embedding(
                num_embeddings=self.hparams["num_embeddings"],
                embedding_dim=embedding_dim,
                padding_idx=PAD_IDX,
            )

        if pretrained_word_bias is not None:
            self.word_bias = nn.Parameter(pretrained_word_bias)
        else:
            self.word_bias = nn.Parameter(torch.zeros(self.hparams["num_embeddings"]))

        if pretrained_context_bias is not None:
            self.context_bias = nn.Parameter(pretrained_context_bias)
        else:
            self.context_bias = nn.Parameter(
                torch.zeros(self.hparams["num_embeddings"])
            )

    def configure_optimizers(self):
        """Adam optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        optim_dict = {"optimizer": optimizer}
        scheduler_dict = {
            "cyclic": torch.optim.lr_scheduler.CyclicLR,
            "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        }
        lr_scheduler_dict = {
            "cyclic": {"interval": "step"},
            "plateau": {"monitor": "train_epoch_loss"},
            None: None,
        }
        scheduler = scheduler_dict[self.hparams["lr_scheduler"]]
        if scheduler:
            if self.hparams["lr_scheduler_args"]:
                scheduler = scheduler(optimizer, **self.hparams["lr_scheduler_args"])
            else:
                scheduler = scheduler(optimizer)
            optim_dict["lr_scheduler"] = {
                **lr_scheduler_dict[self.hparams["lr_scheduler"]],
                **{
                    "scheduler": scheduler,
                    "frequency": 1,
                },
            }
        return optim_dict

    def forward(self, rows: Tensor, cols: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns a tuple of the word and context vectors, respectively."""
        # Input is a tuple of indices
        word_vectors = self.word_embedding(rows)
        context_vectors = self.context_embedding(cols)
        return word_vectors, context_vectors

    def _f_glove(self, co_matrix_elements: Tensor) -> Tensor:
        """The f function from the GloVe paper."""
        condition = co_matrix_elements < self.hparams["x_max"]
        out = torch.where(
            condition,
            (co_matrix_elements / self.hparams["x_max"]) ** self.hparams["alpha"],
            torch.ones_like(co_matrix_elements),
        )
        return out

    def _criterion(
        self,
        word_vectors: Tensor,
        context_vectors: Tensor,
        word_bias: Tensor,
        context_bias: Tensor,
        co_matrix_elements: Tensor,
    ) -> Tensor:
        """The loss function from the GloVe paper. We add a small value to all
        co_matrix_elements before taking the log to regulate zero entries.
        """
        vector_dots = (word_vectors * context_vectors).sum(dim=-1)
        tempered_log = (co_matrix_elements + 1e-16).log()
        f = self._f_glove(co_matrix_elements)
        loss = (vector_dots + word_bias + context_bias - tempered_log) ** 2
        loss = f * loss
        loss = loss.mean(dim=0)
        return loss

    def train_dataloader(self):
        dataset = GloVeDataset(co_matrix_sparse=self.co_matrix_sparse)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["num_workers"],
            persistent_workers=self.hparams["persistent_workers"],
            pin_memory=self.hparams["pin_memory"],
        )
        return loader

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        rows, cols, co_matrix_elements = batch
        word_vectors, context_vectors = self(rows, cols)
        word_bias = self.word_bias[rows]
        context_bias = self.context_bias[cols]
        loss = self._criterion(
            word_vectors, context_vectors, word_bias, context_bias, co_matrix_elements
        )
        self.log("train_batch_loss", loss, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs: Dict[str, Tensor]) -> None:
        """Save to wandb, if save_models_to_wandb."""
        mean_loss = torch.tensor(
            [batch["loss"] for batch in training_step_outputs]
        ).mean()

        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            if self.hparams["save_models_to_wandb"]:
                self.save_model()

        metric_dict = {"train_epoch_loss": mean_loss, "best_loss": self.best_loss}
        for name, metric in metric_dict.items():
            self.log(name, metric, prog_bar=True)

    def save_model(self) -> None:
        """Save state_dict and non-ignored __init__ parameters
        logged to self.hparams to wandb.
        """
        model_file_name = "glove.pt"
        torch.save(self.state_dict(), model_file_name)
        wandb.save(model_file_name)

        # Because co_matrix_sparse was ignored for logging error purposes
        # we need to re-include it before syncing.
        model_init_params_file_name = "model_init_params.pt"
        init_params = self.hparams
        init_params["co_matrix_sparse"] = self.co_matrix_sparse
        torch.save(init_params, model_init_params_file_name)
        wandb.save(model_init_params_file_name)

        print(
            f"Saved at global step: {self.global_step}",
            f"Epoch: {self.current_epoch}",
            f"Loss: {self.best_loss.item()}",
            sep="\n",
        )

    def get_embedding_weights(self) -> Tensor:
        """Returns the mean of the word and context embedding weights."""
        with torch.no_grad():
            word_weights = self.word_embedding.weight.data
            context_weights = self.context_embedding.weight.data
            mean_weights = (word_weights + context_weights) / 2
            return mean_weights.clone().detach()

    def get_embedding_biases(self) -> Tensor:
        """Returns the mean of the word and context biases."""
        with torch.no_grad():
            mean_bias = (self.word_bias + self.context_bias) / 2
            return mean_bias.clone().detach()

    def get_embedding_vector(self, idx: int) -> Tensor:
        """
        Returns the mean of the word and context vectors corresponding to a
        given index.
        """
        with torch.no_grad():
            idx_t = torch.tensor(idx, device=self.device)
            word_vector, context_vector = self(idx_t, idx_t)
            mean_vector = (word_vector + context_vector) / 2
            return mean_vector.clone().detach()

    def get_unit_embedding_vector(self, idx: int) -> Tensor:
        """
        Returns the mean of the word and context vectors with corresponding to
        a given index whose length has been normalized to 1.
        """
        vector = self.get_embedding_vector(idx)
        vector_norm = (vector @ vector).sqrt()
        unit_vector = vector / vector_norm
        return unit_vector.clone().detach()

    def get_embedding_cosine(self, idx1: int, idx2: int) -> Tensor:
        """Returns the cosine between the mean vectors corresponding to the
        provided indices.
        """
        with torch.no_grad():
            unit_vector_1 = self.get_unit_embedding_vector(idx1)
            unit_vector_2 = self.get_unit_embedding_vector(idx2)
            cos = unit_vector_1 @ unit_vector_2
            return cos.clone().detach()

    def get_embedding_dot(self, idx1: int, idx2: int) -> Tensor:
        """Returns the dot product between the mean vectors corresponding
        to the provided indices.
        """
        with torch.no_grad():
            mean_vector_1 = self.get_embedding_vector(idx1)
            mean_vector_2 = self.get_embedding_vector(idx2)
            dot = mean_vector_1 @ mean_vector_2
            return dot.clone().detach()

    def get_word_context_vectors(
        self, rows: Tensor, cols: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Returns the word and context vectors associated with indices
        rows and cols, respectively.
        """
        is_training = self.training
        self.eval()
        with torch.no_grad():
            word_vectors, context_vectors = self(rows, cols)
        if is_training:
            self.train()
        return word_vectors, context_vectors


class CoMatrixBuilder:
    """Class for the vectorized generation of sparse co-occurrence matrices,
    as defined in GloVe, built with pytorch.

    Description
    ----------
    Encoding performed using a word_to_idx_dict dict mapping words to indices.
    <PAD> and <UNK> are expected to map to 0 and 1, respectively.
    Text is expected to have been passed through text_normalizer first.
    Text normalization and proper mapping via word_to_idx_dict can be verified
    through the check_normalization flag.

    The default behavior is to weight words in the context window by factors of
    1 / (distance-to-center-word) as in GloVe. This can be toggled to constant
    weighting through the glove_window_weighting flag.

    Parameters
    ----------
    text : str
        Text to to be embedded
    tokens_df : DataFrame
        Token counts data stored in 'word' and 'count' columns.
    min_word_count : int, default 1
        Minimum count for a word in tokens_df to be included in the vocabulary.
    context_window : int, default 2
        Width of the context window used on either side of the center word.
    batch_size : int, default 128
        Dataloader batch_size arg.
    device : torch.device default torch.device('cpu')
        Device for tensors during co-matrix construction.
    num_workers : int, default 0
        Dataloader num_workers arg.
    pin_memory : bool, default False
        Dataloader pin_memory arg.
    non_blocking : bool, default False
        arg for tensor.to calls when passing to cuda.
    glove_window_weighting : bool, default True
        Flag for weighing context of center word with 1 / n decay,
        as in GloVe. False weighs all context words equally.
    include_center_in_context : bool, default False
        Experimental flag for including the center word in its own context.
    perform_symmetry_sanity_check : bool, default True
        Flag for verifying that the co-occurrence matrix is symmetric, up to
        tolerances from text edges. A very memory-consuming check.

    Methods
    ----------
    generate_co_matrix :
        Populates the dense co-occurrence tensor, stored as
    the `co_matrix` attr.
    get_sparse_co_matrix :
        Returns the co_matrix as a sparse tensors and optionally writes to disk
        at `saved_path`.
    """

    def __init__(
        self,
        text: str,
        tokens_df: DataFrame,
        min_word_count: int = 1,
        context_window: int = 2,
        batch_size: int = 128,
        device: torch.device = torch.device("cpu"),
        num_workers: int = 0,
        pin_memory: bool = False,
        non_blocking: bool = False,
        glove_window_weighting: bool = True,
        include_center_in_context: bool = False,
        check_normalization: bool = True,
        perform_symmetry_sanity_check: bool = True,
    ) -> None:
        super().__init__()

        self.text = text
        self.context_window = context_window
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.non_blocking = non_blocking
        self.glove_window_weighting = glove_window_weighting
        self.include_center_in_context = include_center_in_context
        self.perform_symmetry_sanity_check = perform_symmetry_sanity_check

        # Get num_embeddings from tokens
        self.tokens_df = deepcopy(tokens_df)
        if min_word_count > 1:
            if "count" in self.tokens_df:
                self.tokens_df = self.tokens_df[
                    self.tokens_df["count"] >= min_word_count
                ]
            else:
                warnings.warn(
                    "count column does not exist in tokens_df DataFrame, min_word_count arg ignored."
                )

        self.word_to_idx_dict = word_to_idx_dict_from_df(self.tokens_df)
        self.vocab_size = len(self.word_to_idx_dict)

        self._is_co_matrix_generated = False
        self.co_matrix = torch.zeros(
            self.vocab_size, self.vocab_size, dtype=torch.int64, device=self.device
        )

        """Use int64's to avoid accumulated floating-point errors. If
        glove_window_weighting, the right-context window weights are taken to be
        (context_window, context_window -1, ...) in order to use int64
        everywhere. In this case, the co_matrix will be divided by
        context_window beore returning, effectively making the weights
        (1, 1/2, 1/3, ...) as desired.

        Using int64 appears to hugely boost the speed of the computation, also,
        but this is possibly very hardward-dependent:
        https://stackoverflow.com/questions/2550281/floating-point-vs-integer-calculations-on-modern-hardware
        """
        if glove_window_weighting:
            left_window_weights_list = list(range(1, context_window + 1))
        else:
            left_window_weights_list = [1 for _ in range(context_window)]
        right_window_weights_list = left_window_weights_list[::-1]

        if include_center_in_context:
            window_weights_list = (
                left_window_weights_list + [context_window] + right_window_weights_list
            )
        else:
            window_weights_list = left_window_weights_list + right_window_weights_list
        self.window_weights = torch.tensor(window_weights_list, device=self.device)
        # right_window_weights_list used in symmetry check performed on cpu.
        self.right_window_weights = torch.tensor(right_window_weights_list)

        self._dataset = CoMatrixDataset(
            text=self.text,
            word_to_idx_dict=self.word_to_idx_dict,
            context_window=self.context_window,
            include_center_in_context=self.include_center_in_context,
        )

        self._dataloader = DataLoader(
            dataset=self._dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def _get_sparse_window_weights(self, context: Tensor) -> Tensor:
        batch_size = context.shape[0]
        sparse_window_weights = torch.zeros(
            batch_size, self.vocab_size, dtype=torch.int64, device=self.device
        )
        expanded_window_weights = self.window_weights.expand(*context.shape)
        sparse_window_weights.scatter_add_(1, context, expanded_window_weights)
        return sparse_window_weights

    def _fill_co_matrix(self, center_idx: Tensor, context: Tensor) -> None:
        sparse_window_weights = self._get_sparse_window_weights(context)
        self.co_matrix.index_add_(0, center_idx, sparse_window_weights)

    def _seconds_to_readable(self, seconds: int) -> str:
        """Helper for converting seconds to more readable times."""
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)

        time_names = ("days", "hours", "mins", "secs")
        time_vals = (days, hours, mins, secs)
        time_list = []
        for name, val in zip(time_names, time_vals):
            if val:
                time_list.append(f"{val} {name}")
        time_str = " ".join(time_list)
        return time_str

    def generate_co_matrix(self) -> None:
        """Populates the dense self.co_matrix tensor."""
        assert (
            not self._is_co_matrix_generated
        ), "Co-occurrence matrix has already been generated. Call get_sparse_co_matrix()."
        self._is_co_matrix_generated = True
        num_batches = len(self._dataloader)
        progress_dict = {num_batches // 10 * n: n / 10.0 for n in range(1, 10)}
        print("Generating co-occurrence matrix...")
        start_time = perf_counter()
        for batch_idx, (center_idx, context) in enumerate(self._dataloader):
            if batch_idx in progress_dict:
                frac_complete = progress_dict[batch_idx]
                current_runtime = perf_counter() - start_time
                seconds_remaining = int(current_runtime * (1 / frac_complete - 1))
                print(
                    f"{int(frac_complete * 100)}% complete.",
                    f"Approx. {self._seconds_to_readable(seconds_remaining)} remaining.",
                )
            center_idx = center_idx.to(self.device, non_blocking=self.non_blocking)
            context = context.to(self.device, non_blocking=self.non_blocking)
            self._fill_co_matrix(center_idx, context)
        print("...done!")

        if self.perform_symmetry_sanity_check:
            max_asymm = torch.max(self.co_matrix - self.co_matrix.T)
            right_window_weights_max_count = torch.arange(
                1, self.right_window_weights.shape[0] + 1, dtype=torch.int64
            )
            max_possible_asymm = (
                2 * self.right_window_weights @ right_window_weights_max_count
            )
            if max_asymm > max_possible_asymm:
                warnings.warn(
                    "Sanity check failed: matrix is not symmetric (within edge-effects tolerance)"
                )
                print(
                    f"Co-occurrence matrix asymmetry: {max_asymm}",
                    f"Max asymmetry tolerance: {max_possible_asymm.item()}",
                    sep="\n",
                )
        # Fix normalization of GloVe weight window, if glove_window_weighting.
        if self.glove_window_weighting:
            self.co_matrix = self.co_matrix / self.context_window

    def get_sparse_co_matrix(self, save_path: Optional[str] = None):
        """Returns a sparse version of the generated self.co_matrix tensor
        and optionally writes the result to save_path.
        """
        assert (
            self._is_co_matrix_generated
        ), "Co-occurrence matrix has not been generated, call generate_co_matrix() first."
        sparse_co_matrix = self.co_matrix.to(torch.device("cpu")).to_sparse().detach()
        if save_path is not None:
            torch.save(sparse_co_matrix, save_path)
        return sparse_co_matrix
