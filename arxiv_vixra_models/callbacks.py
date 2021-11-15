from . import *

class WandbTextCallback(pl.Callback):
    """Callback class for visualizing text predictions.

    Description
    ----------
    Takes in the one-hot-encoded inputs, targets, and plain text and logs the
    model's predictions to wandb.

    `__init__` Args
    ----------
    `val_sample`: `Tuple[torch.Tensor]`
        Tuple of (inputs, target) taken from the validation set.
    `decoder_fn`: Callable
        Function to be called on architecture inputs as in
        decoder_fn(input, translation_data), returning plain text.
    `translation_data`: str or pd.DataFrame
        DataFrame object or path to feather file containing data required for
        decoder_fn to translate architecture inputs.
    `labels`: Sequence[str], optional, default = ('0', '1')
        Class labels.
    `name`: str, optional, default = None
        Name to print on ROC and PR plots. Using the default (None) prints
        "Classifier" as the name.
    `table_samples`: int, optional, default = 10
        Number of examples to display in the wandb table.
    """
    def __init__(self,
                 val_sample: torch.Tensor,
                 decoder_fn: Callable,
                 translation_data: Union[str, DF],
                 labels: Optional[Sequence[str]] = ('0', '1'),
                 table_samples: Optional[int] = 10,
                 name: Optional[Union[None, str]] = None) -> None:
        super().__init__()
        self.val_sample_inputs, self.val_sample_targets = val_sample
        self.val_sample_text = [decoder_fn(input, translation_data) for input in self.val_sample_inputs]
        self.labels = labels
        self.name = name
        if len(self.val_sample_inputs) < table_samples:
            self.table_samples = len(self.val_sample_inputs)
        else:
            self.table_samples = table_samples
        

    def on_validation_epoch_end(self, trainer: pl.Trainer, 
                          pl_module: pl.LightningModule) -> None:
        val_tensors = self.val_sample_inputs.to(device=pl_module.device)
        # Must still use .eval and no_grad context in pl inference.
        pl_module.eval()
        with torch.no_grad():
            preds, _ = pl_module(val_tensors)
        # Move to CPU. Not sure if I need to detach?
        preds = preds.detach().cpu()
        targets = self.val_sample_targets.detach().cpu()
        # Manipulate data to expected np arrays.
        y_true = np.array([n.item() for n in targets])
        y_preds = np.array([n.item() for n in preds])
        y_preds_int = np.rint(y_preds)
        
        # Table for visualizing predictions.
        results_dic = {'text': self.val_sample_text,
                       'prediction': y_preds,
                       'source': y_true}
        results_df = pd.DataFrame.from_dict(results_dic).iloc[:self.table_samples]
        example_table = wandb.Table(dataframe=results_df)
        
        # Histogram of prediction distribution.
        histogram = wandb.Histogram(preds)
        # Precision-recall, ROC, and confusion matrix
        # sklearn summary plots
        cm = ConfusionMatrixDisplay.from_predictions(y_true,
                                        y_preds_int, 
                                        display_labels=self.labels,
                                        normalize='true')
        roc = RocCurveDisplay.from_predictions(y_true,
                                 y_preds,
                                 name=self.name)
        pr = PrecisionRecallDisplay.from_predictions(y_true,
                                        y_preds,
                                        name=self.name)
        # Use more informative labels.
        roc.ax_.set_xlabel('False Positive Rate')
        roc.ax_.set_ylabel('True Positive Rate')
        roc.ax_.set_title(f'ROC Curve (Positive class: {self.labels[1]})')
        pr.ax_.set_xlabel('Recall')
        pr.ax_.set_ylabel('Precision')
        pr.ax_.set_title(f'Precision-Recall Curve (Positive class: {self.labels[1]})')
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
