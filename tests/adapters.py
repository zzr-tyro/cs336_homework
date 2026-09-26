from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import regex as re
import numpy as np


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    return torch.matmul(in_features, weights.T)

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, "..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    #在weights中查询向量可以使用矩阵当作索引，此时会将输入矩阵的每一个人元素替换成weights里面的一个嵌入向量，完全不用关心shape
    return weights[token_ids]
    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """

    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    """
    在低维空间（4096 维）中，很多复杂的特征是混杂、缠绕在一起的（就像一盘纠缠不清的红蓝线绳）。

    升维（升到 11008 维）：把特征投影到更高的维度空间。在更高维的空间里，原本缠绕在一起的特征更容易被线性切割开。

    SiLU 门控过滤：把不需要的特征通道“关掉”（乘以接近 0 的门值），把有用的特征“放大”。

    降维（降回 4096 维）：把高维空间里提炼好的“纯净特征”，重新压缩投影回 4096 维的标准接口，传给下一层。
    """

    w1 = torch.matmul(in_features, w1_weight.T)
    w1_silu = run_silu(w1)
    w3 = torch.matmul(in_features, w3_weight.T)

    return torch.matmul(w1_silu * w3, w2_weight.T)
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    scale = Q.shape[-1] ** 0.5
    attn = torch.matmul(Q, K.transpose(-1, -2)) / scale
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))

    scores = run_softmax(attn, -1)
    return torch.matmul(scores, V)

    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    seq_len = in_features.shape[-2]

    Q = run_linear(d_model, d_model, q_proj_weight, in_features)
    K = run_linear(d_model, d_model, k_proj_weight, in_features)
    V = run_linear(d_model, d_model, v_proj_weight, in_features)

    Q = Q.view(Q.shape[: -1] + (num_heads, d_model // num_heads)).transpose(-3, -2)
    K = K.view(K.shape[: -1] + (num_heads, d_model // num_heads)).transpose(-3, -2)
    V = V.view(V.shape[: -1] + (num_heads, d_model // num_heads)).transpose(-3, -2)

    # casual mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    attn_out = run_scaled_dot_product_attention(Q, K, V, mask)  # B, nh, T, d_v // nh

    ## 先交换维度，再通过 contiguous 刷新内存以便后续合并多头, 因为veiw函数需要张量连续存放
    attn_out = attn_out.transpose(-3, -2)
    attn_out = attn_out.contiguous()
    attn_out = attn_out.view(attn_out.shape[:-2] + (d_model,))

    return run_linear(d_model, d_model, o_proj_weight, attn_out)
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    seq_len = in_features.shape[-2]

    #rope位置旋转编码与普通位置编码不一样，其尽在计算注意力分数时应用，也就是说仅对QK使用
    Q = run_linear(d_model, d_model, q_proj_weight, in_features)
    K = run_linear(d_model, d_model, k_proj_weight, in_features)
    V = run_linear(d_model, d_model, v_proj_weight, in_features)

    Q = Q.view(Q.shape[: -1] + (num_heads, d_model // num_heads)).transpose(-3, -2)
    K = K.view(K.shape[: -1] + (num_heads, d_model // num_heads)).transpose(-3, -2)
    V = V.view(V.shape[: -1] + (num_heads, d_model // num_heads)).transpose(-3, -2)

    if token_positions is not None:
        Q = run_rope(d_model // num_heads, theta, max_seq_len, Q, token_positions)
        K = run_rope(d_model // num_heads, theta, max_seq_len, K, token_positions)

    # casual mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    attn_out = run_scaled_dot_product_attention(Q, K, V, mask)  # B, nh, T, d_v // nh

    ## 先交换维度，再通过 contiguous 刷新内存以便后续合并多头, 因为veiw函数需要张量连续存放
    attn_out = attn_out.transpose(-3, -2)
    attn_out = attn_out.contiguous()
    attn_out = attn_out.view(attn_out.shape[:-2] + (d_model,))

    return run_linear(d_model, d_model, o_proj_weight, attn_out)
    raise NotImplementedError


"""
输入向量 [..., S, d_k]
       │
       ▼
1. 拆分成 2D 向量对：(x0, x1), (x2, x3), ..., (x_{d-2}, x_{d-1})
       │
       ▼
2. 计算旋转频率 θ_i = 1 / (theta ** (2i / d_k))
       │
       ▼
3. 计算每个位置 m 的角度: m * θ_i
       │
       ▼
4. 应用旋转公式: 
   x_0_new =  x_0 * cos(mθ) - x_1 * sin(mθ)
   x_1_new =  x_0 * sin(mθ) + x_1 * cos(mθ)
       │
       ▼
输出向量 [..., S, d_k]
"""
def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """

    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
    freqs = token_positions.unsqueeze(-1).float() * freqs

    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

    rotate_in_query_or_key = torch.stack((-1 * in_query_or_key[..., 1::2], in_query_or_key[..., 0::2]), dim=-1).flatten(-2)

    result = (in_query_or_key * cos) + (rotate_in_query_or_key * sin)

    return result


    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    #transform模块
    batch_size, sequence_length, _ = in_features.size()
    token_positions = torch.arange(sequence_length, device=in_features.device).expand(batch_size, -1)

    ln1 = run_rmsnorm(d_model,1e-5, weights["ln1.weight"], in_features)
    attn_out = run_multihead_self_attention_with_rope(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        ln1,
        token_positions,
    )

    x1 = attn_out + in_features #残差连接

    ln2 = run_rmsnorm(d_model,1e-5, weights["ln2.weight"], x1)
    ffn_out = run_swiglu(
        d_model,
        d_ff,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        ln2
    )

    return ffn_out + x1
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    #transform前向传播
    #embedding
    #batch_size, sequence_length = in_indices.size()
    token_embeddings = run_embedding(vocab_size, d_model, weights["token_embeddings.weight"], in_indices)

    #解码器区域
    hidden_vector = token_embeddings
    for i in range(num_layers):
        #构建当前层的权重
        layer_weights = {
            k[len(f'layers.{i}.'):]: v
            for k, v in weights.items()
            if k.startswith(f'layers.{i}.')
        }

        #运用集成的transformer
        hidden_vector = run_transformer_block(
            d_model,
            num_heads,
            d_ff,
            context_length,
            rope_theta,
            layer_weights,
            hidden_vector
        )

    #输出层全连接
    hidden_vector = run_rmsnorm(d_model, 1e-6, weights['ln_final.weight'], hidden_vector)

    # 语言模型头部 - 将隐藏状态映射到词汇表大小
    lm_logits = run_linear(vocab_size, d_model, weights['lm_head.weight'], hidden_vector)

    return lm_logits
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, "d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    square = in_features ** 2
    rms = torch.sqrt(torch.mean(square, dim=-1, keepdim=True) + eps)
    return in_features / rms * weights

    raise NotImplementedError


def run_silu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    #对所有元素使用silu(x / (1 + exp(-x)))

    deno = 1.0 + torch.exp(-in_features)
    return in_features / deno

    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    #重点要求随机采样这样每次的batch都会不一样，训练的数据也会不一样
    starts = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,)
    )

    inputs = torch.stack([
        torch.tensor(dataset[start : start + context_length])
        for start in starts
    ])
    labels = torch.stack([
        torch.tensor(dataset[start + 1 : start + 1 + context_length])
        for start in starts
    ])

    return inputs, labels
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of **(with the same shape)** as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    #return torch.softmax(in_features, dim)
    max_num = torch.max(in_features, dim = dim, keepdim = True).values
    exp_features = torch.exp(in_features - max_num)
    sum_features = torch.sum(exp_features, dim = dim, keepdim = True)

    return exp_features / sum_features
    raise NotImplementedError

#在工程上需要对softmax负对数进行数学变换才能更好求取loss值
def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_num = torch.max(inputs, dim=-1, keepdim=True).values
    exp_sum_inputs = torch.sum(torch.exp(inputs - max_num), dim = -1, keepdim=True)

    # 1. 把 targets 的形状从 [batch_size] 扩展为 [batch_size, 1]，以匹配 inputs 的维度数
    targets_expanded = targets.unsqueeze(-1)  # 形状变为 [batch_size, 1]
    # 2. 沿着最后一个维度 (dim=-1) 按照索引提取值 形状为 [batch_size, 1]
    target_logits = inputs.gather(
        dim=-1, index=targets_expanded
    )

    loss = torch.log(exp_sum_inputs) + max_num - target_logits
    return loss.mean()
    raise NotImplementedError


#梯度剪切正则化，仅算grad, 是其所有grad参量的平方和开方后的值小于max_l2_norm
def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    filter_nograd_parameters = [p for p in parameters if p.grad is not None]

    if len(filter_nograd_parameters) == 0:
        return

    now_scale = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in filter_nograd_parameters))
    zoom_scale = max_l2_norm / (now_scale + 1e-8)

    if now_scale > max_l2_norm:
        for p in filter_nograd_parameters:
            p.grad.data.mul_(zoom_scale)

    #raise NotImplementedError

"""
AdamW (自定义优化器实例，优化器内部结构)
│
├── self.defaults (全局默认参数字典)
│    ├── 'lr': 0.001
│    ├── 'betas': (0.9, 0.999)
│    ├── 'eps': 1e-8
│    └── 'weight_decay': 0.01
│
├── self.state (字典: 记录每个参数的历史 Momentum/状态)
│    │  # 键(Key)是参数 p 的内存地址/引用，值(Value)是对应的状态字典
│    ├── p0: {                                 #这里的p0与p1与下面group0里面params中的p0与p1对应
│    │      'step': 10,                            # 当前更新到第几步                           
│    │      'exp_avg': Tensor([...]),              # m_t 一阶矩 (动量)
│    │      'exp_avg_sq': Tensor([...])            # v_t 二阶矩 (梯度平方)
│    │   }
│    └── p1: {
│    │      'step': 10,                            # 当前更新到第几步
│    │      'exp_avg': Tensor([...]),              # m_t 一阶矩 (动量)
│    │      'exp_avg_sq': Tensor([...])            # v_t 二阶矩 (梯度平方)
│    │   }
│
└── self.param_groups (列表: 包含一个或多个超参数组字典)
     │
     └── [ Group 0 字典 ]
          ├── 'lr': 0.001                         # 当前组的学习率
          ├── 'betas': (0.9, 0.999)               # 当前组的 betas
          ├── 'eps': 1e-8                         # 当前组的 eps
          ├── 'weight_decay': 0.01                # 当前组的 weight_decay
          │
          └── 'params': [ p0, p1, p2, ... ]       # 存放【Tensor 句柄对象】的列表
               │
               ├── p0  <=== (这就是代码里的 for p in group['params'] 拿到的对象)
               │    │
               │    ├── [属性 1] .data ---------> 指向 C++ 底层的【权重数值内存块】 (Weights)
               │    │                                存放模型此时此刻真正的参数，如: [[0.25, -0.11], ...]
               │    │
               │    ├── [属性 2] .grad ---------> 指向 C++ 底层的【梯度数值内存块】 (Grads)
               │    │                                执行 loss.backward() 后填入，如: [[0.01, -0.002], ...]
               │    │                                (未跑 backward 前为 None)
               │    │
               │    ├── [属性 3] .requires_grad -> True (标记此 Tensor 是否需要自动求导)
               │    ├── [属性 4] .grad_fn ------> <AddBackward0 ...> (反向传播计算图的节点指针)
               │    ├── [属性 5] .device -------> device(type='cuda', index=0) (数据存储在 CPU 还是 GPU)
               │    └── [属性 6] .dtype --------> torch.float32 (数据精度类型)
               │
               ├── p1 (下一个层的 Tensor 对象，内部结构与 p0 完全一致)
               └── p2 (下一个层的 Tensor 对象)
               """

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 0.001, betas = (0.9, 0.99), eps = 1e-8, weight_decay: float = 1e-2):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not eps >= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        #超参数字典
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure = None) -> float:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                state = self.state[p]
                #如果第一次没有历史值需要新建0向量与0step
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['eps']
                grad = p.grad.data #加上.data表示仅仅提取数值，可以免于梯度计算的追踪

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                step_size = lr * ((1 - beta2  ** state["step"]) ** 0.5) / (1 - beta1 ** state["step"])
                div = exp_avg_sq.sqrt().add_(eps)

                p.data.addcdiv_(exp_avg, div, value = -step_size)

        return loss

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW
    raise NotImplementedError

import math
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it < cosine_cycle_iters: #余弦下降使用0到pi半个周期的形状，而不是0到pi/2四分之一个周期的形状
        return (max_learning_rate - min_learning_rate) * 0.5 * (1.0 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) + min_learning_rate
    else:
        return min_learning_rate


    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }

    if isinstance(out, (str, os.PathLike)):
        with open(out, 'wb') as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)

    #raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """

    if isinstance(src, (str, os.PathLike)):
        with open(src, 'rb') as f:
            checkpoint = torch.load(f)
    else:
        checkpoint = torch.load(src)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']

    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)

    raise NotImplementedError

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.merges_dict = {pair: rank for rank, pair in enumerate(merges)}

    @staticmethod
    def _word_to_bytes_list(word: str) -> list[bytes]:
        l = list(word.encode("utf-8"))
        l = [bytes([x]) for x in l]
        return l

    #根据特殊标记划分段落，对每个段落根据预分词规则划分出一个一个word
    def _partition(self, text: str) -> list[list[list[bytes]]]:
        if len(self.special_tokens) != 0:
            self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            chunks = re.split("(" + "|".join(map(re.escape, self.special_tokens)) + ")", text)
        else:
            chunks = [text]

        new_chunks = []
        for chunk in chunks:
            if chunk == "":
                continue
            # special token
            if chunk in self.special_tokens:
                new_chunks.append(
                    [
                        [
                            chunk.encode("utf-8")
                        ]
                    ]
                )
                continue
            new_chunk = []
            for match in self.PAT.finditer(chunk):
                word = match.group(0)
                list_match = self._word_to_bytes_list(word)
                new_chunk.append(list_match)

            new_chunks.append(new_chunk)

        return new_chunks

    #每一个单独的word进行按照merge规则合并处理
    def _merge_word(self, word):

        while True:
            min_rank = len(self.merges)
            merge_pos = None

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.merges_dict.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    merge_pos = i

            if merge_pos is None:
                break

            word = (word[:merge_pos] + [word[merge_pos] + word[merge_pos + 1]] + word[merge_pos + 2:])

        return word

    #将chunks里面的每一个chunk里面的每个单词合并
    def _merge(self, chunks: list[list[list[bytes]]]) -> list[list[list[bytes]]]:
        new_chunks = []
        for chunk in chunks:

            new_chunk = []
            for word in chunk:
                new_chunk.append(self._merge_word(word))

            new_chunks.append(new_chunk)

        return  new_chunks

    #根据合并的结果在逆向词表查询id，并返回id的列表
    def encode(self, text: str) -> list[int]:
        chunks = self._merge(self._partition(text))
        tokens = []
        for chunk in chunks:
            for word in chunk:
                for token in word:
                    id = self.reverse_vocab[token]
                    tokens.append(id)

        return tokens



    #解码函数
    def decode(self, ids: list[int]) -> str:
        # 1. 把所有 token 对应的 bytes 片段拼成一个完整的 bytes 对象
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)

        # 2. 对完整的字节流进行 UTF-8 解码
        return full_bytes.decode("utf-8", errors="replace")

    #流式编码
    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        for chunk in iterable:
            yield from self.encode(chunk)

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special all_tokens).
        special_tokens (list[str]): A list of string special all_tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple all_tokens, and will always be
            kept as a single token. If these special all_tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = dict()
    merges = []
    #初始化字典，前面的位置空出让给特殊字符，后面的位置给正常的256字符编码
    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])
    now_size = 256 + len(special_tokens)
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode("utf-8")

    #预处理
    #读取文本
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    #特殊字符分割
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    #词频统计加预分词
    # GPT-2 标准预分词正则表达式
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def to_bytes_tuple(word: str) -> tuple[bytes, ...]:
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)

    #构建词频统计的列表
    pre_tokens_cnt = {}
    for chunk in chunks:
        for match in PAT.finditer(chunk):
            word = match.group(0)
            key = to_bytes_tuple(word)
            if pre_tokens_cnt.get(key) is None:
                pre_tokens_cnt[key] = 1
            else:
                pre_tokens_cnt[key] += 1


    #合并字符，统计频次更新字典与更新tokens列表
    while now_size < vocab_size:
        save = dict()
        for tokens, cnt in pre_tokens_cnt.items():
            i = 0
            while i < len(tokens) - 1:
                token = (tokens[i], tokens[i + 1])
                if save.get(token) is None:
                    save[token] = cnt
                else:
                    save[token] += cnt
                i = i + 1

        #寻找最高频次的组合
        maxfre = max(save.values())
        candidates = [k for k, v in save.items() if v == maxfre]
        best_token = max(candidates)

        #更新字典与合并规则
        a, b = best_token
        vocab[now_size] = a + b
        merges.append((a, b))
        now_size += 1

        #更新pre_tokens_cnt
        #收集变更
        changes = []
        for old_tokens, cnt in pre_tokens_cnt.items():
            mark = 0
            i = 0
            new_tokens = []
            while i < len(old_tokens):
                if i < len(old_tokens) - 1 and old_tokens[i] == a and old_tokens[i + 1] == b:
                    new_tokens.append (old_tokens[i] + old_tokens[i + 1])
                    i += 2
                    mark = 1
                else:
                    new_tokens.append (old_tokens[i])
                    i += 1
            if mark:
                new_tokens = tuple(new_tokens)
                changes.append((new_tokens, old_tokens, cnt))

        #改变pre_tokens_cnt
        for new_tokens, old_tokens, cnt in changes:
            pre_tokens_cnt[new_tokens] = cnt
            del pre_tokens_cnt[old_tokens]

    #print(merges)
    return vocab, merges
    #raise NotImplementedError

