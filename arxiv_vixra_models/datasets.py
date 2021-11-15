from . import *
from .one_hot import one_hot_encoding
from .embedding import string_to_ints

class OneHotCharDataset(Dataset):
    """
    Dataset subclass for processing text in feather/csv files at character level.

    Description
    ----------
    
    Assumes each row in the data feather file has 'source' column which
    is either 'arxiv' or 'vixra' and that a separate feather file mapping characters
    to integers (1-indexing assumed) exists. Outputs one-hot-encoded characters.

    Args
    ----------
    `data`: str or pd.DataFrame
        DataFrame object or path to data feather file.
    `chars`: str or pd.DataFrame
        DataFrame object or path to character feather file.
    `text_column`: str
        Column in data feather file containing text, e.g. `'title'`
    or `'title'`.
    `seq_len`: int
        How many characters are read in from the text.
    `sample_size`: int or float or None, optional, default=None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    `check_normalization`: bool, optional, default = True
        Check whether the text data was normalized according to text_normalizer.

    Notable Methods 
    ----------
    `__getitem__`
        Returns `(text_tensor, source_int)`  where `text_tensor` is the
        one-hot encoded text and `source_int` is  0, 1 if from `'arxiv'`,
         `'vixra'`, respectively, both pytorch tensors.
     """
    
    def __init__(self,
                 data: Union[str, DF],
                 chars: Union[str, DF],
                 text_column: str,
                 seq_len: int,
                 sample_size: Optional[Union[int, float, None]] = None,
                 check_normalization: Optional[bool] = True) -> None:
        super().__init__()
        if isinstance(data, str):
            self.data = pd.read_feather(data, columns=[text_column, 'source'])
        else:
            # Important to copy, otherwise data input is modified.
            self.data = data.copy()
        if isinstance(chars, str):
            self.chars_df = pd.read_feather(chars)
        else:
            self.chars_df = chars
        self._char_to_idx = dict(zip(self.chars_df['char'], self.chars_df['idx']))
        self.text_column = text_column
        self.seq_len = seq_len
        if sample_size is not None:
            # .copy() needed in below steps to avoid onnx export errors.
            if isinstance(sample_size, float):
                self.data = self.data.sample(frac=sample_size).copy()
            if isinstance(sample_size, int):
                self.data = self.data.sample(sample_size).copy()
        self.check_normalization = check_normalization

        # Perform one-hot encoding and mapping source to 0/1 for arxiv/vixra.

        self.data['source'] = self.data['source'].apply(self._arxiv_vixra_to_int)
        self.data[text_column] = self.data[text_column].apply(self._one_hot_encoding)
        
    def _arxiv_vixra_to_int(self, s: str) -> Tensor:
        if s == 'vixra':
            return torch.tensor(True)
        if s == 'arxiv':
            return torch.tensor(False)
        raise ValueError('Source string must either be arxiv or vixra, invalid data.' )
        
    def _one_hot_encoding(self, s: str) -> Tensor:
        return one_hot_encoding(s, self._char_to_idx, self.seq_len, self.check_normalization)
        
    def __len__(self) -> int:
        return len(self.data)
        

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        source = self.data.iloc[idx].loc['source']
        one_hot_text = self.data.iloc[idx].loc[self.text_column]

        return one_hot_text, source
        
class EmbeddingDataset(Dataset):
    """
    Dataset subclass for processing text in feather/csv files at word level.

    Description
    ----------
    
    Assumes each row in the data feather file has 'source' column which
    is either 'arxiv' or 'vixra' and that a separate feather file mapping words
    to integers (1-indexing assumed) exists. Outputs integer-encoded words.

    Args
    ----------
    `data`: str or pd.DataFrame
        DataFrame object or path to data feather file.
    `vocab`: str or pd.DataFrame
        DataFrame object or path to vocabulary feather file.
    `text_column`: str
        Column in data feather file containing text, e.g. `'title'`
    or `'title'`.
    `seq_len`: int
        Maximum number of words included from the text in a sequence.
    `sample_size`: int or float or None, optional, default=None
        If not None, include only a specific number of data points (if int) or
        specific fraction of data (if float), randomly chosen from the data.
        Applied separately to each training set: train, valid, and test.
    `check_normalization`: bool, optional, default = True
        Check whether the text data was normalized according to text_normalizer.

    Notable Methods 
    ----------
    `__getitem__`
        Returns `(text_tensor, source_int)`  where `text_tensor` is the
        integer-encoded text and `source_int` is  0, 1 if from `'arxiv'`,
         `'vixra'`, respectively, both pytorch tensors.
     """
    
    def __init__(self,
                 data: Union[str, DF],
                 vocab: Union[str, DF],
                 text_column: str,
                 seq_len: int,
                 sample_size: Optional[Union[int, float, None]] = None,
                 check_normalization: Optional[bool] = True) -> None:
        super().__init__()
        if isinstance(data, str):
            self.data = pd.read_feather(data, columns=[text_column, 'source'])
        else:
            # Important to copy, otherwise data input is modified.
            self.data = data.copy()
        if isinstance(vocab, str):
            self.vocab_df = pd.read_feather(vocab)
        else:
            self.vocab_df = vocab
        # Create dictionary from vocab. Start from 2, accounting for padding and
        # unknown maps at 0 and 1.
        self._word_to_idx = dict(zip(self.vocab_df['word'], np.arange(2, len(self.vocab_df) + 2)))
        
        self.text_column = text_column
        self.seq_len = seq_len
        if sample_size is not None:
            # .copy() needed in below steps to avoid onnx export errors.
            if isinstance(sample_size, float):
                self.data = self.data.sample(frac=sample_size).copy()
            if isinstance(sample_size, int):
                self.data = self.data.sample(sample_size).copy()
        self.check_normalization = check_normalization

        # Perform embedding and map source to 0/1 for arxiv/vixra.
        self.data['source'] = self.data['source'].apply(self._arxiv_vixra_to_int)
        self.data[text_column] = self.data[text_column].apply(self._string_to_ints)
        
    def _arxiv_vixra_to_int(self, s: str) -> Tensor:
        if s == 'vixra':
            return torch.tensor(True)
        if s == 'arxiv':
            return torch.tensor(False)
        raise ValueError('Source string must either be arxiv or vixra, invalid data.' )
        
    def _string_to_ints(self, s: str) -> Tensor:
        return string_to_ints(s, self._word_to_idx, self.seq_len, self.check_normalization)
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        source = self.data.iloc[idx].loc['source']
        tokenized_text = self.data.iloc[idx].loc[self.text_column]

        return tokenized_text, source