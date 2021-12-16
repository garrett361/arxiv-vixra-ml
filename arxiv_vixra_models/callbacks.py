import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import torch
from typing import Tuple, Sequence, Optional, Union, Callable
import wandb


class WandbTextCallback(pl.Callback):
    """Callback class for visualizing text predictions.

    Description
    ----------
    Takes in the one-hot-encoded inputs, targets, and plain text and logs the
    model's predictions to wandb.

    `__init__` Args
    ----------
    `val_sample`: `Tuple[torch.Tensor, torch.Tensor]`
        Tuple of (inputs, target) taken from the validation set.
    `decoder_fn`: Callable
        Function to be called on architecture inputs as in
        decoder_fn(input, translation_data), returning plain text.
    `translation_data`: str or DataFrame
        DataFrame object or path to feather file containing data required for
        decoder_fn to translate architecture inputs.
    `labels`: Sequence[str], optional, default = ('0', '1')
        Class labels.
    `name`: str, optional, default = None
        Name to print on ROC and PR plots. Using the default (None) prints
        "Classifier" as the name.
    `table_samples`: int, optional, default = None
        Number of examples to display in the wandb table.  Default shows all.
    """

    def __init__(self,
                 val_sample: Tuple[torch.Tensor, torch.Tensor],
                 decoder_fn: Callable,
                 translation_data: Union[str, pd.DataFrame],
                 labels: Optional[Sequence[str]] = ('0', '1'),
                 table_samples: Optional[int] = None,
                 name: Optional[Union[None, str]] = None) -> None:
        super().__init__()
        self.val_sample_inputs, self.val_sample_targets = val_sample
        # Only strip *leading* white space as a sanity check. There shouldn't be
        # any trailing whitespace ever.
        self.val_sample_text = [decoder_fn(
            input, translation_data).lstrip() for input in self.val_sample_inputs]
        self.labels = labels
        self.name = name
        if table_samples is None:
            self.table_samples = len(self.val_sample_inputs)
        else:
            self.table_samples = min(
                table_samples, len(self.val_sample_inputs))

    def on_validation_epoch_end(self,
                                trainer: pl.Trainer,
                                pl_module: pl.LightningModule) -> None:
        val_tensors = self.val_sample_inputs.to(device=pl_module.device)
        # Assume there is an inference methods which simply returns the probs.
        # Reshape to same structure as targets.
        probs = pl_module.inference(val_tensors).view(-1)
        # Move to CPU. Not sure if I need to detach?
        probs = probs.detach().cpu()
        targets = self.val_sample_targets.detach().cpu()
        # Manipulate data to expected np arrays.
        y_true = np.array([int(n.item()) for n in targets])
        y_probs = np.array([n.item() for n in probs])
        y_probs_int = np.rint(y_probs)

        # Table for visualizing predictions.
        results_dic = {'text': self.val_sample_text,
                       'prediction': y_probs,
                       'source': y_true}
        results_df = pd.DataFrame.from_dict(
            results_dic).iloc[:self.table_samples]
        # Reorder by least confident to most confident.
        example_table = wandb.Table(dataframe=results_df.sort_values('prediction',
                                                                     key=lambda x: abs(x - 1 / 2))
                                    )
        # Histogram of prediction distribution.
        histogram = wandb.Histogram(probs)
        # Precision-recall, ROC, and confusion matrix
        # sklearn summary plots
        cm = ConfusionMatrixDisplay.from_predictions(y_true,
                                                     y_probs_int,
                                                     display_labels=self.labels,
                                                     normalize='true')
        roc = RocCurveDisplay.from_predictions(y_true,
                                               y_probs,
                                               name=self.name)
        pr = PrecisionRecallDisplay.from_predictions(y_true,
                                                     y_probs,
                                                     name=self.name)
        # Use more informative labels.
        roc.ax_.set_xlabel('False Positive Rate')
        roc.ax_.set_ylabel('True Positive Rate')
        roc.ax_.set_title(f'ROC Curve (Positive class: {self.labels[1]})')
        pr.ax_.set_xlabel('Recall')
        pr.ax_.set_ylabel('Precision')
        pr.ax_.set_title(
            f'Precision-Recall Curve (Positive class: {self.labels[1]})')
        # Increase quality by saving first with dpi specified, then uploading.
        cm.figure_.savefig("cm.jpg", dpi=250)
        roc.figure_.savefig("roc.jpg", dpi=250)
        pr.figure_.savefig("pr.jpg", dpi=250)

        # Log all metrics in sync with the global step.
        wandb.log({"examples": example_table,
                   "histogram": histogram,
                   "confusion matrix": wandb.Image("cm.jpg"),
                   "roc curve": wandb.Image("roc.jpg"),
                   "precision recall": wandb.Image("pr.jpg"),
                   "global_step": trainer.global_step})
