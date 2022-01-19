import numpy as np
from pandas import DataFrame
import pytorch_lightning as pl
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from torch import Tensor
from torch.nn import Module
from typing import Tuple, Sequence, Optional, Callable, Dict
import wandb

from .embedding_utils import (
    embedding_cosine_heatmap,
    pca_3d_embedding_plotter_topk,
    tsne_3d_embedding_plotter_topk,
)


class WandbAVPredictionCallback(pl.Callback):
    """Callback class for visualizing arxiv/vixra classsification
    predictions.

    Description
    ----------
    Logs images of the confusion matrix, ROC, and PR curves at
    each step, in addition to sorted predictions for individual
    text samples and a histogram of the predicted probabilities.
    Assumes the model either has a get_probs or get_probs_hiddens
    method from which we can get probability predictions.

    Parameters
    ----------
    val_sample : Tuple[Tensor, Tensor]
        Tuple of (inputs, targets) taken from the validation set.
    decoder_fn : Callable
        Function to be called on architecture inputs as in
        decoder_fn(input, decoder_dict), returning plain text.
    decoder_dict : dict
        Mapping used for decoding text with decoder_fn
    labels : Sequence[str], default ('arxiv', 'vixra')
        Class labels.
    name : str, default None
        Name to print on ROC and PR plotsl, if any.
    table_samples : int, default None
        Number of examples to display in the wandb table.  Default shows all.
    """

    def __init__(
        self,
        val_sample: Tuple[Tensor, Tensor],
        decoder_fn: Callable,
        decoder_dict: Dict[int, str],
        labels: Optional[Sequence[str]] = ("arxiv", "vixra"),
        table_samples: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.val_sample_inputs, self.val_sample_targets = val_sample
        # Only strip *leading* white space as a sanity check. There shouldn't
        # be any trailing whitespace ever.
        self.val_sample_text = [
            decoder_fn(input, decoder_dict).lstrip() for input in self.val_sample_inputs
        ]
        self.labels = labels
        self.name = name
        if table_samples is None:
            self.table_samples = len(self.val_sample_inputs)
        else:
            self.table_samples = min(table_samples, len(self.val_sample_inputs))

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val_tensors = self.val_sample_inputs.to(device=pl_module.device)
        if hasattr(pl_module, "get_probs_hiddens"):
            probs, _ = pl_module.get_probs_hiddens(val_tensors)
            probs = probs.view(-1)
        else:
            probs = pl_module.get_probs(val_tensors).view(-1)
        probs = probs.cpu()
        targets = self.val_sample_targets.cpu()
        # Manipulate data to expected np arrays.
        y_true = np.array([int(n.item()) for n in targets])
        y_probs = np.array([n.item() for n in probs])
        y_probs_int = np.rint(y_probs)

        # Table for visualizing individual text predictions.
        examples_dict = {
            "text": self.val_sample_text,
            "prediction": y_probs,
            "source": y_true,
        }
        examples_df = DataFrame(examples_dict).iloc[: self.table_samples]
        # Order by least confident to most confident.
        examples_table = wandb.Table(
            dataframe=examples_df.sort_values(
                "prediction", key=lambda x: abs(x - 1 / 2)
            )
        )
        # Histogram of prediction distribution.
        histogram = wandb.Histogram(probs)
        # Precision-recall, ROC, and confusion matrix
        # sklearn summary plots
        cm = ConfusionMatrixDisplay.from_predictions(
            y_true, y_probs_int, display_labels=self.labels, normalize="true"
        )
        roc = RocCurveDisplay.from_predictions(y_true, y_probs, name=self.name)
        pr = PrecisionRecallDisplay.from_predictions(y_true, y_probs, name=self.name)
        # Use more informative labels.
        roc.ax_.set_xlabel("False Positive Rate")
        roc.ax_.set_ylabel("True Positive Rate")
        roc.ax_.set_title(f"ROC Curve (Positive class: {self.labels[1]})")
        pr.ax_.set_xlabel("Recall")
        pr.ax_.set_ylabel("Precision")
        pr.ax_.set_title(f"Precision-Recall Curve (Positive class: {self.labels[1]})")

        cm.figure_.savefig("cm.png", bbox_inches="tight")
        roc.figure_.savefig("roc.png", bbox_inches="tight")
        pr.figure_.savefig("pr.png", bbox_inches="tight")

        # Log all metrics in sync with the global step.
        wandb.log(
            {
                "examples": examples_table,
                "histogram": histogram,
                "confusion matrix": wandb.Image("cm.png"),
                "roc curve": wandb.Image("roc.png"),
                "precision recall": wandb.Image("pr.png"),
                "global_step": trainer.global_step,
            }
        )


class WandbVisualEmbeddingCallback(pl.Callback):
    """Callback class for visualizing embeddings.

    Description
    ----------
    Generates a heatmap, PCA, and t-SNE visualizations of embeddings.

    Parameters
    ----------
    model : torch.nn.Module
        Model used to compute cosines.
    heatmap_words : list of strs
        List of words for heatmap visualization.
    pca_words : list of strs
        List of words for PCA visualization.
    tsne_words : list of strs
        List of words for t-SNE visualization.
    word_to_idx_dict : dict
        Mapping from words to indices.
    idx_to_word_dict : dict
        Mapping from indices to words.
    k : int, default 5
        Top-k closest words are plotted in PCA and t-SNE.
    pca_args : dict, optional
        Optional args to be passed to PCA.
    tsne_args : dict, optional
        Optional args to be passed to t-SNE.
    heatmap_title : str, optional
        Optional title for heatmap visualization.
    pca_title : str, optional
        Optional title for PCA visualization.
    tsne_title : str, optional
        Optional title for t-SNE visualization.
    """

    def __init__(
        self,
        model: Module,
        heatmap_words: Sequence[str],
        pca_words: Sequence[str],
        tsne_words: Sequence[str],
        word_to_idx_dict: Dict[str, int],
        idx_to_word_dict: Dict[int, str],
        k: int = 5,
        pca_args: Optional[Dict] = None,
        tsne_args: Optional[Dict] = None,
        heatmap_title: Optional[str] = None,
        pca_title: Optional[str] = None,
        tsne_title: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.heatmap_words = heatmap_words
        self.pca_words = pca_words
        self.tsne_words = tsne_words
        self.word_to_idx_dict = word_to_idx_dict
        self.idx_to_word_dict = idx_to_word_dict
        self.k = k
        self.pca_args = pca_args
        self.tsne_args = tsne_args
        self.heatmap_title = heatmap_title
        self.pca_title = pca_title
        self.tsne_title = tsne_title

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        heatmap = embedding_cosine_heatmap(
            model=self.model,
            words=self.heatmap_words,
            word_to_idx_dict=self.word_to_idx_dict,
            title=self.heatmap_title,
        )
        pca = pca_3d_embedding_plotter_topk(
            model=self.model,
            words=self.pca_words,
            word_to_idx_dict=self.word_to_idx_dict,
            idx_to_word_dict=self.idx_to_word_dict,
            k=self.k,
            title=self.pca_title,
            pca_args=self.pca_args,
        )
        tsne = tsne_3d_embedding_plotter_topk(
            model=self.model,
            words=self.tsne_words,
            word_to_idx_dict=self.word_to_idx_dict,
            idx_to_word_dict=self.idx_to_word_dict,
            k=self.k,
            title=self.tsne_title,
            tsne_args=self.tsne_args,
        )

        heatmap.savefig("heatmap.png", bbox_inches="tight")

        # Log all metrics in sync with the global step.
        wandb.log(
            {
                "heatmap": wandb.Image("heatmap.png"),
                "pca": wandb.Plotly(pca),
                "tsne": wandb.Plotly(tsne),
                "global_step": trainer.global_step,
            }
        )


class WandbTextGenerationCallback(pl.Callback):
    """Callback class for visualizing the text generated by language models.

    Description
    ----------
    Expects pl_module to have a generate_text(seed, text_char_len, topk),
    signature indicated, which generates text of character length
    text_char_len starting from some initial seed str and which draws
    from the topk predictions of the model when generating.

    Parameters
    ----------
    seeds : sequence of strs, default ['quantum', 'field', 'theory']
        Sequence of strs which are used to seed the generation.
        len(seeds) samples will visualized by the callback.
    text_char_len : int, default 256
        Character length of generated text.
    topk : int, default 3
        Text is generated by drawing from the topk predictions
        of the model.
    """

    def __init__(
        self,
        seeds: Sequence[str] = ["quantum", "field", "theory"],
        text_char_len: int = 256,
        topk: int = 3,
    ) -> None:
        super().__init__()
        self.seeds = seeds
        self.text_char_len = text_char_len
        self.topk = topk

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        generated_text = []
        for seed in self.seeds:
            generated_text.append(
                pl_module.generate_text(seed, self.text_char_len, self.topk)
            )
        generated_text_dict = {"generated text": generated_text, "seed": self.seeds}
        generated_text_df = DataFrame(generated_text_dict)
        generated_text_table = wandb.Table(dataframe=generated_text_df)
        # Log in sync with the global step.
        wandb.log(
            {"generated text": generated_text_table, "global_step": trainer.global_step}
        )
