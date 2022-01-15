from copy import deepcopy
from typing import Tuple, Optional, Dict, Sequence

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import Module

from .text_normalizer import text_normalizer

PAD_IDX = 0
UNK_IDX = 1
BLANK_IDX = 0


def str_to_idxs(
    s: str,
    word_to_idx: Dict[str, int],
    seq_len: Optional[int] = None,
    check_normalization: bool = True,
    strip_before_normalization_check: bool = False,
) -> torch.Tensor:
    """Encode a string `s` as an integer tensor.

    Description
    ----------
    Encoding performed using a word_to_idx dict mapping words to indices.
    <PAD> and <UNK> are expected to map to 0 and 1, respectively. Optionally
    force the text to be seq_len long.  Text is expected to have been passed
    through text_normalizer first. Text normalization and proper mapping via
    word_to_idx can be verified through the check_normalization flag.

    Args
    ----------
    `s`: str
        Text to to be embedded
    `word_to_idx`: dict
        Mapping from words to indices.
    `seq_len`: int or None, default = 'None'
        Force text to be of length seq_len, if not None.
    `check_normalization`: bool, default = False
        Verify the text and word_to_idx mapping are of the proper formats.
    `strip_before_normalization_check`: bool, default = False
        Flag for whether to strip text before performing normalization check.

    Returns
    ----------
    `s_int_tensor`: torch.Tensor
        Integer tensor.
    """
    if check_normalization:
        s_check = s.strip() if strip_before_normalization_check else s
        assert s_check == text_normalizer(
            s_check
        ), "String not normalized as expected, apply text_normalizer first."
        pad_check = word_to_idx.get("<PAD>", None)
        assert (
            pad_check == PAD_IDX
        ), f"word_to_idx is expected to map <PAD> to {PAD_IDX}, not {pad_check}."
        unk_check = word_to_idx.get("<UNK>", None)
        assert (
            unk_check == UNK_IDX
        ), f"word_to_idx is expected to map <UNK> to {UNK_IDX}, not {unk_check}."

    # Forcing sequence to be seq_len words long, if seq_len is not None.
    s_list = s.split()
    s_len = len(s_list)
    if seq_len is None:
        seq_len = s_len
    if seq_len > s_len:
        padding = (seq_len - s_len) * [PAD_IDX]
        s_int_list = padding + [word_to_idx.get(w, UNK_IDX) for w in s_list]
    else:
        s_int_list = [word_to_idx.get(w, UNK_IDX) for w in s_list[:seq_len]]
    s_int_tensor = torch.tensor(s_int_list)
    return s_int_tensor


def idxs_to_str(
    tensor: torch.Tensor,
    idx_to_word: Dict[int, str],
    check_normalization: bool = False,
    replace_pad_with_space: bool = True,
) -> str:
    """Decodes an integer tensor to a string.

    Description
    ----------
    Decoding performed using a idx_to_word dict mapping indices to words.
    <PAD> and <UNK> are assumed to map to 0 and 1, respectively.
    Proper keys in idx_to_word can be verified through the check_normalization
    flag. By default <PAD> is mapped to blank space for aesthetics.

    Args
    ----------
    `s`: str
        Text to to be embedded
    `idx_to_word`: dict
        Mapping from indices to words.
    `check_normalization`: bool, default = False
        Verify the idx_to_word mapping keys have the proper form.
    `replace_pad_with_space`: bool,  default = True
        Flag for replacing <PAD> with blank space.

    Returns
    ----------
    `text`: str
        Decoded string.
    """
    if check_normalization:
        pad_check = idx_to_word.get(PAD_IDX, None)
        assert (
            pad_check == "<PAD>"
        ), f"idx_to_word is expected to map {PAD_IDX} to <PAD>, not {pad_check}."
        idx_check = idx_to_word.get(UNK_IDX, None)
        assert (
            idx_check == "<UNK>"
        ), f"idx_to_word is expected to map {UNK_IDX} to <UNK>, not {idx_check}."
    if replace_pad_with_space:
        idx_to_word = deepcopy(idx_to_word)
        idx_to_word[PAD_IDX] = " "
    text = " ".join(idx_to_word[n.item()] for n in tensor)
    return text


def embedding_cosine_heatmap(
    model: Module,
    words: Sequence[str],
    word_to_idx: Dict[str, int],
    title: Optional[str] = None,
) -> Figure:
    """Returns a heatmap of the cosines between vectors corresponding to words.

    Description
    ----------
    Encoding performed using a word_to_idx dict mapping words to indices.
    <PAD> and <UNK> are expected to map to 0 and 1, respectively. model is
    assumed to use the same word-idx mappings for its embedding layer. model
    is also assumed to have a get_embedding_cosine(idx1, idx2) method for
    computing the corresponding cosines.

    Args
    ----------
    `model`: torch.nn.Module
        Model used to compute cosines.
    `words`: list of strs
        List of words to be compared
    `word_to_idx`: dict
        Mapping from words to indices.
    `title`: str
        Optional plot title.

    Returns
    ----------
    Matplotlib Figure object.
    """
    idxs = [word_to_idx[word] for word in words]
    sorted_words_idxs = sorted(zip(words, idxs), key=lambda x: x[1])
    sorted_words, sorted_idxs = list(zip(*sorted_words_idxs))

    def idx_cosines(idx):
        return [
            model.get_embedding_cosine(idx, other_idx).cpu().item()
            for other_idx in sorted_idxs
        ]

    cos_dict = {word: idx_cosines(idx) for word, idx in sorted_words_idxs}
    cos_df = pd.DataFrame(cos_dict, index=sorted_words)
    sns.set(font_scale=1.5)
    _, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cos_df,
        ax=ax,
        annot=True,
        fmt=".2f",
        vmax=1.0,
        vmin=0.0,
        xticklabels=cos_df.columns,
        yticklabels=cos_df.index,
        annot_kws={"size": 8},
    )
    if title is not None:
        ax.set_title(title)
    return ax.figure


def topk_cosines(
    model: Module, words: Sequence[str], word_to_idx: Dict[str, int], k: int = 5
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Returns torch topk tensor for closest words to all words in `words`,
    as measured by cosine.

    Description
    ----------
    Encoding performed using a word_to_idx dict mapping words to indices.
    <PAD> and <UNK> are expected to map to 0 and 1, respectively. model is
    assumed to use the same word-idx mappings for its embedding layer. model
    is also assumed to have a get_embedding_weights method for which returns
    the embedding matrix.

    Args
    ----------
    `model`: torch.nn.Module
        Model used to compute cosines.
    `words`: list of strs
        List of words to be compared
    `word_to_idx`: dict
        Mapping from words to indices.
    `k`: int, default = 3
        k value for topk

    Returns
    ----------
    torch topk results, a named Tuple of tensors
    """
    words_idx = torch.tensor([word_to_idx[word] for word in words], device=model.device)
    embeddings = model.get_embedding_weights()
    embeddings_norms = (embeddings ** 2).sum(dim=-1).sqrt()[:, None]
    unit_embeddings = embeddings / embeddings_norms
    # Norming makes <PAD> direction nan, which topk determines to be larger
    # than any float. Fix:
    unit_embeddings[0] = 0.0
    word_vector = unit_embeddings[words_idx]
    cosines = word_vector @ unit_embeddings.T
    topk_cosines = cosines.topk(k, dim=-1)
    return topk_cosines


def topk_cosines_vectors(
    model: Module, words: Sequence[str], word_to_idx: Dict[str, int], k: int = 5
) -> torch.Tensor:
    """Returns topk vectors for closest words to all words in `words`,
    as measured by cosine.

    Description
    ----------
    Encoding performed using a word_to_idx dict mapping words to indices.
    <PAD> and <UNK> are expected to map to 0 and 1, respectively. model is
    assumed to use the same word-idx mappings for its embedding layer. model
    is also assumed to have a get_embedding_weights method for which returns
    the embedding matrix.

    Each word is included amongst its own top-k closest words, as this is useful
    for plotting functions.

    Args
    ----------
    `model`: torch.nn.Module
        Model used to compute cosines.
    `words`: list of strs
        List of words to be compared
    `word_to_idx`: dict
        Mapping from words to indices.
    `k`: int, default = 3
        k value for topk

    Returns
    ----------
    torch.Tensor
    """
    topk_word_cosines = topk_cosines(model, words, word_to_idx, k)
    topk_word_cosines_idxs = topk_word_cosines.indices
    embeddings = model.get_embedding_weights()
    topk_vectors = embeddings[topk_word_cosines_idxs]
    return topk_vectors


def topk_cosines_df(
    model: Module,
    words: Sequence[str],
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    k: int = 5,
) -> pd.DataFrame:
    """Returns pd.DataFrame containing data on the k-closest words for all elements of
    `words`, as measured by cosine.

    Description
    ----------
    Encoding performed using a word_to_idx dict mapping words to indices.
    <PAD> and <UNK> are expected to map to 0 and 1, respectively. idx_to_word
    is the inverse mapping. model is assumed to use the same word-idx mappings
    for its embedding layer. model is also assumed to have a
    get_embedding_weights method for which returns the embedding matrix.

    The returned pd.DataFrame has the following columns:
        -'seed': contains words provided in `words`.
        -'word': contains instances of words close to 'seed' entry.
        -'idx': the idx of the 'word' column entry.
        -'cos': cosine between 'seed' and 'word' vectors.


    Args
    ----------
    `model`: torch.nn.Module
        Model used to compute cosines.
    `words`: list of strs
        List of words to be compared.
    `word_to_idx`: dict
        Mapping from words to indices.
    `idx_to_word`: dict
        Mapping from indices to words.
    `k`: int, default = 3
        k value for topk.

    Returns
    ----------
    pd.DataFrame
    """
    topk_word_cosines = topk_cosines(model, words, word_to_idx, k)
    topk_word_cosines_idxs = topk_word_cosines.indices.cpu().numpy()
    topk_word_cosines_cos = topk_word_cosines.values.cpu().numpy()

    def idx_to_word_fn(idx):
        return idx_to_word[idx]

    idx_to_word_fn_vectorized = np.vectorize(idx_to_word_fn)
    topk_words = idx_to_word_fn_vectorized(topk_word_cosines_idxs)
    topk_seeds = idx_to_word_fn_vectorized(topk_word_cosines_idxs[:, 0])

    seed = np.concatenate([[word] * k for word in topk_seeds])
    word = np.concatenate(topk_words)
    word_idx = np.concatenate(topk_word_cosines_idxs)
    cos = np.concatenate(topk_word_cosines_cos)

    topk_dict = {"seed": seed, "word": word, "word_idx": word_idx, "cos": cos}
    topk_df = pd.DataFrame(topk_dict)
    return topk_df


def topk_analogies(
    model: Module,
    analogy_str_seq: Sequence[str],
    word_to_idx: Dict[str, int],
    k: int = 5,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Returns topk results for the top-k words which best complete
    the analogy word0 : word1 :: word2 : ???.

    Description
    ----------
    If analogy_str_seq = (word0, word1, word2) and we denote the associated
    vectors with the same symbols, the precise computation is to find the word
    word3 which has the largest cosine with the sum word1 - word0 + word2; see
    https://nlp.stanford.edu/pubs/glove.pdf page 6.

    Encoding performed using a word_to_idx dict mapping words to indices.
    <PAD> and <UNK> are expected to map to 0 and 1, respectively. model is
    assumed to use the same word-idx mappings for its embedding layer. model
    is also assumed to have a get_embedding_weights method for which returns
    the embedding matrix.

    Args
    ----------
    `model`: torch.nn.Module
        Model used to compute cosines.
    `analogy_str_seq`: sequence of three strs
        Tuple, e.g., of the form (word0, word1, word2) for analogy
        word0 : word1 :: word2 : ???.
    `word_to_idx`: dict
        Mapping from words to indices.
    `k`: int, default = 3
        k value for topk.

    Returns
    ----------
    torch topk results, a named Tuple of tensors.
    """
    analogy_idxs = torch.tensor(
        [word_to_idx[word] for word in analogy_str_seq], device=model.device
    )
    embeddings = model.get_embedding_weights()
    embeddings_norms = (embeddings ** 2).sum(dim=-1).sqrt()[:, None]
    unit_embeddings = embeddings / embeddings_norms
    # Norming makes <PAD> direction nan, which topk determines to be larger
    # than any float. Fix:
    unit_embeddings[0] = 0.0

    # The first three vectors are only normed after they are summed up.
    analogy_vectors = embeddings[analogy_idxs]
    analogy_vectors[0] = -1 * analogy_vectors[0]
    analogy_vectors_sum = analogy_vectors.sum(dim=0)
    analogy_vectors_sum_normed = (
        analogy_vectors_sum / (analogy_vectors_sum @ analogy_vectors_sum).sqrt()
    )
    analogy_cosines = analogy_vectors_sum_normed @ unit_embeddings.T
    topk_analogy_cosines = analogy_cosines.topk(k, dim=-1)
    return topk_analogy_cosines


def topk_analogies_df(
    model: Module,
    analogy_str_seq: Tuple[str, str, str],
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    k: int = 5,
) -> pd.DataFrame:
    """Returns pd.DataFrame containing data on the top-k words which best
    complete the analogy word0 : word1 :: word2 : ???.

    Description
    ----------
    Encoding performed using a word_to_idx dict mapping words to indices.
    <PAD> and <UNK> are expected to map to 0 and 1, respectively. idx_to_word
    is the inverse mapping. model is assumed to use the same word-idx mappings
    for its embedding layer. model is also assumed to have a
    get_embedding_weights method for which returns the embedding matrix.

    The returned pd.DataFrame has the following columns:
        -'word0': First word in the analogy.
        -'word1': Second word in the analogy.
        -'word2': Third word in the analogy.
        -'word3': Predicted final in the analogy.
        -'word3_idx': the idx of the 'word3' column entry.
        -'cos': cosine between word0 - word1 + word2 and the word3 vector.

    We filter out the three-words in analogy_str_seq from the top-k results.

    Args
    ----------
    `model`: torch.nn.Module
        Model used to compute cosines.
    `analogy_str_seq`: tuple of three strs
        Tuple (word0, word1, word2) for analogy word0 : word1 :: word2 : ???.
    `word_to_idx`: dict
        Mapping from words to indices.
    `idx_to_word`: dict
        Mapping from indices to words.
    `k`: int, default = 3
        k value for topk.

    Returns
    ----------
    pd.DataFrame
    """

    topk_analogy_results = topk_analogies(model, analogy_str_seq, word_to_idx, k + 3)
    topk_analogy_results_idxs = topk_analogy_results.indices.cpu().numpy()
    topk_analogy_results_cos = topk_analogy_results.values.cpu().numpy()

    def idx_to_word_fn(idx):
        return idx_to_word[idx]

    idx_to_word_fn_vectorized = np.vectorize(idx_to_word_fn)
    topk_analogy_results_words = idx_to_word_fn_vectorized(topk_analogy_results_idxs)

    words_idxs_cos_zip = zip(
        topk_analogy_results_words, topk_analogy_results_idxs, topk_analogy_results_cos
    )
    filtered_words_idxs_cos = np.array(
        [
            (word, idx, cos)
            for word, idx, cos in words_idxs_cos_zip
            if word not in analogy_str_seq
        ]
    )
    filtered_words, filtered_idxs, filtered_cos = filtered_words_idxs_cos[:k].T

    topk_dict = {
        "word0": [analogy_str_seq[0]] * k,
        "word1": [analogy_str_seq[1]] * k,
        "word2": [analogy_str_seq[2]] * k,
        "word3": filtered_words,
        "word3_idx": filtered_idxs,
        "cos": filtered_cos,
    }
    topk_df = pd.DataFrame(topk_dict)
    return topk_df


def px_word_df_3d(
    df: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    z: str = "z",
    color: str = "seed",
    symbol: str = "seed",
    text: str = "word",
    size: str = "cos",
    title: Optional[str] = None,
) -> Figure:
    """Generates 3D plots of data generated by topk_cosines_df, after appending
    3D spatial location column data.

    Description
    ----------
    pd.DataFrame is expected to have x, y, z data in similarly named columns.
    The 'seed' and 'cos' columns are used to determine markers.

    Returns
    ----------
    plotly Figure object.
    """
    fig = px.scatter_3d(
        df, x=x, y=y, z=z, color=color, symbol=symbol, text=text, size=size, title=title
    )
    # Hide axis labels and ticks.
    hide_axis_dict = {"tickfont": {"color": "rgba(0,0,0,0)"}, "title": {"text": ""}}
    fig.update_layout(
        scene={
            "xaxis": hide_axis_dict,
            "yaxis": hide_axis_dict,
            "zaxis": hide_axis_dict,
        }
    )
    return fig


def pca_3d_embedding_plotter_topk(
    model: Module,
    words: Sequence[str],
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    k: int = 5,
    title: Optional[str] = None,
    pca_args: Optional[Dict] = None,
) -> Figure:
    """Generates a PCA plot of the elements of `words` and their top-k closest
    neighbors by cosine for each element. model is also assumed to have a
    get_embedding_weights method for which returns the embedding matrix.

    Returns
    ----------
    Matplotlib Figure object.
    """
    topk_words_df = topk_cosines_df(model, words, word_to_idx, idx_to_word, k)
    word_idx_array = np.array(topk_words_df["word_idx"])
    word_vectors = model.get_embedding_weights().cpu().numpy()[word_idx_array]

    pca_args_dict = {"n_components": 3}
    if pca_args is not None:
        pca_args_dict = {**pca_args_dict, **pca_args}
    word_vectors_scaled = StandardScaler().fit_transform(word_vectors)
    word_vectors_scaled_pca = PCA(**pca_args_dict).fit_transform(word_vectors_scaled)

    topk_words_df["x"] = word_vectors_scaled_pca[:, 0]
    topk_words_df["y"] = word_vectors_scaled_pca[:, 1]
    topk_words_df["z"] = word_vectors_scaled_pca[:, 2]

    return px_word_df_3d(topk_words_df, title=title)


def tsne_3d_embedding_plotter_topk(
    model: Module,
    words: Sequence[str],
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    k: int = 5,
    title: Optional[str] = None,
    tsne_args: Optional[Dict] = None,
) -> Figure:
    """Generates a t-SNE plot of the elements of `words` and their top-k closest
    neighbors by cosine for each element. model is also assumed to have a
    get_embedding_weights method for which returns the embedding matrix.

    Returns
    ----------
    Matplotlib Figure object.
    """
    topk_words_df = topk_cosines_df(model, words, word_to_idx, idx_to_word, k)
    word_idx_array = np.array(topk_words_df["word_idx"])
    word_vectors = model.get_embedding_weights().cpu().numpy()[word_idx_array]

    # t-SNE defaults set by experimentation.
    tsne_args_dict = {
        "n_components": 3,
        "perplexity": 5,
        "init": "pca",
        "n_jobs": -1,
        "learning_rate": 10,
        "n_iter": 10000,
    }
    if tsne_args is not None:
        tsne_args_dict = {**tsne_args_dict, **tsne_args}
    word_vectors_tsne = TSNE(**tsne_args_dict).fit_transform(word_vectors)

    topk_words_df["x"] = word_vectors_tsne[:, 0]
    topk_words_df["y"] = word_vectors_tsne[:, 1]
    topk_words_df["z"] = word_vectors_tsne[:, 2]

    return px_word_df_3d(topk_words_df, title=title)