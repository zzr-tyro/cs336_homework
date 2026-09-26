from __future__ import annotations

import os
import random
from typing import Any, Callable, Literal

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase



def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    bench_size = len(prompt_strs)
    max_len = 0
    raw_input_ids = []
    raw_labels = []
    raw_response_masks = []
    for i in range(bench_size):
        prompt_id = tokenizer.encode(prompt_strs[i], add_special_tokens=False)
        output_id = tokenizer.encode(output_strs[i], add_special_tokens=False)
        total = prompt_id + output_id
        max_len = max(len(total) - 1, max_len)

        input_ids = total[0: len(total) - 1]
        raw_input_ids.append(input_ids)
        labels = total[1: len(total)]
        raw_labels.append(labels)

        response_mask = [0] * (len(prompt_id) - 1) + [1] * len(output_id)
        raw_response_masks.append(response_mask)

    # 从tokenizer库中得到padding的id
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    # 预填充，然后替换
    padded_input_ids = torch.full(
        (bench_size, max_len), pad_token_id, dtype=torch.long
    )
    padded_labels = torch.full(
        (bench_size, max_len), pad_token_id, dtype=torch.long
    )
    padded_response_masks = torch.zeros((bench_size, max_len), dtype=torch.long)

    for i in range(bench_size):
        seq_len = len(raw_input_ids[i])
        # 显式使用 torch.tensor 进行类型转换赋值
        padded_input_ids[i, :seq_len] = torch.tensor(
            raw_input_ids[i], dtype=torch.long
        )
        padded_labels[i, :seq_len] = torch.tensor(
            raw_labels[i], dtype=torch.long
        )
        padded_response_masks[i, :seq_len] = torch.tensor(
            raw_response_masks[i], dtype=torch.long
        )

    return {"input_ids": padded_input_ids, "labels": padded_labels, "response_mask": padded_response_masks}
    raise NotImplementedError


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    # 前向传播获取下一个词元的整个列表
    outputs = model(input_ids)
    logits = outputs.logits
    log_logits = F.log_softmax(logits, dim=-1)
    # gather函数专门用于按索引提取
    log_probs = torch.gather(log_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        softmax_logits = F.softmax(logits, dim=-1)
        # H = - sum(P * logP)
        token_entropy = -torch.sum(softmax_logits * log_logits, dim=-1)
        return {"log_probs": log_probs, "token_entropy": token_entropy}

    return {"log_probs": log_probs}
    # raise NotImplementedError


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    rollout_batch_size = len(rollout_responses)
    total_rewards = torch.empty(rollout_batch_size, dtype=torch.float)
    format_rewards = torch.empty(rollout_batch_size, dtype=torch.float)
    for i in range(rollout_batch_size):
        result = reward_fn(rollout_responses[i], repeated_ground_truths[i])
        total_rewards[i] = result["reward"]
        format_rewards[i] = result["format_reward"]
    metadata = {"mean_total_reward": total_rewards.mean().item(), "mean_format_reward": format_rewards.mean().item()}
    return total_rewards, metadata
    # raise NotImplementedError


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    grouped_rewards = raw_rewards.view(-1, group_size)
    # 计算组内均值
    if baseline == "mean":
        mean = grouped_rewards.mean(dim=-1, keepdim=True)
        advantages = grouped_rewards - mean
    elif baseline == "none":
        advantages = grouped_rewards.clone()
    else:
        raise NotImplementedError(f"Unsupported baseline: {baseline}")

    # 计算组内标准差并归一化
    if advantage_normalizer == "std":
        # 使用torch.std
        std = grouped_rewards.std(dim=-1, keepdim=True)
        advantages = advantages / (std + advantage_eps)
    elif advantage_normalizer == "mean":
        mean = grouped_rewards.mean(dim=-1, keepdim=True)
        advantages = advantages / (mean + advantage_eps)
    elif advantage_normalizer == "none":
        pass
    else:
        raise NotImplementedError(f"Unsupported normalizer: {advantage_normalizer}")

    # 展平回一维向量
    advantages = advantages.view(-1)

    # 记录 metadata
    metadata = {
        "reward_mean": float(raw_rewards.mean()),
        "reward_std": float(raw_rewards.std()),
        "reward_max": float(raw_rewards.max()),
        "reward_min": float(raw_rewards.min()),
    }

    return advantages, metadata
    # raise NotImplementedError


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, # 其形状是（batch_size, 1）不用再进行unsqueeze了
    policy_log_probs: torch.Tensor, # 已经是对数概率
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    metadata = {}
    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = (-1) * raw_rewards_or_advantages * policy_log_probs
        return per_token_policy_gradient_loss, metadata

    elif importance_reweighting_method == "noclip":
        ratio = torch.exp(policy_log_probs - old_log_probs)
        per_token_policy_gradient_loss = (-1) * raw_rewards_or_advantages * ratio
        return per_token_policy_gradient_loss, metadata

    # grpo要求悲观估计，在正常与不截断的二者中选择后loss绝对值小的那一个
    elif importance_reweighting_method == "grpo":
        ratio = torch.exp(policy_log_probs - old_log_probs)
        per_token_policy_gradient_loss1 = -1 * raw_rewards_or_advantages * ratio
        ratio = torch.clamp(ratio, min = 1-cliprange, max = 1+cliprange)
        per_token_policy_gradient_loss2 = -1 * raw_rewards_or_advantages * ratio
        per_token_policy_gradient_loss = torch.max(per_token_policy_gradient_loss1, per_token_policy_gradient_loss2)
        return per_token_policy_gradient_loss, metadata

    # 计算整个回复序列的平均对数比率（Average Log-Ratio），并将其作为整个序列统一的重加权权重。
    elif importance_reweighting_method == "gspo":
        log_ratio = policy_log_probs - old_log_probs
        masked_log_ratio = log_ratio * response_mask
        sequence_length = log_ratio.size(-1)
        # 因为是统计response_token中没有被padding的数的均值，所以不能直接mean，而是要先求和，再除以response_token中1的总数
        response_token_counts = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        ratio = torch.exp(torch.sum(masked_log_ratio, dim = -1, keepdim = True) / response_token_counts)
        ratio = ratio.repeat(1, sequence_length)

        per_token_policy_gradient_loss1 = -1 * raw_rewards_or_advantages * ratio
        ratio = torch.clamp(ratio, min = 1-cliprange, max= 1+cliprange)
        per_token_policy_gradient_loss2 = -1 * raw_rewards_or_advantages * ratio

        per_token_policy_gradient_loss = torch.max(per_token_policy_gradient_loss1, per_token_policy_gradient_loss2)

        # 在真实的gspo中最后这行代码需要写上并运行，这个作业将统计总的loss放在后面的函数中了，所以乘以mask的工作也放在后面的函数里了
        # per_token_policy_gradient_loss = per_token_policy_gradient_loss * response_mask
        # metadata 在测试的时候没有测试，真正的实现需要根据需要填写
        return per_token_policy_gradient_loss, metadata
    raise NotImplementedError


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    per_token_policy_gradient_loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        seq_token_counts = mask.sum(dim=-1).clamp(min=1.0)
        seq_mean_loss = torch.sum(per_token_policy_gradient_loss,dim = -1) / seq_token_counts
        loss = torch.mean(seq_mean_loss)
        return loss
    elif loss_normalization == "constant":
        total_loss = torch.sum(per_token_policy_gradient_loss)
        loss = total_loss / normalization_constant
        return loss
    else:
        raise NotImplementedError
    # raise NotImplementedError


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str], # 输入的提示词
    rollout_responses: list[str], # 旧模型输出的答案
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    raw_rewards, reward_metadata = run_compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    raw_rewards_or_advantages, adv_metadata = run_compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer
    )
    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps
    loss = torch.zeros(0)
    loss_metadata = {}
    for i in range(0, len(repeated_prompts), microbatch_size):
        inputs_microbatch = repeated_prompts[i:i + microbatch_size]
        labels_microbatch = rollout_responses[i:i + microbatch_size]
        bpe = run_tokenize_prompt_and_output(inputs_microbatch, labels_microbatch, tokenizer)
        log_prob_microbatch = run_get_response_log_probs(model, bpe["input_ids"], bpe["labels"], False)["log_probs"]
        """
            全量 Tokenize 与 微批次 Tokenize 的 Padding 差异
            1. old_log_probs 是在函数外部【全量 Batch】一次性算出来的。
            在传入 run_grpo_train_step 之前，测试用例在外部对全部的 repeated_prompts 和 rollout_responses进行了一次全量 Tokenize。
            所以按照整个 batch 的最长 seq 来填充
            2. log_prob_microbatch 是在函数内部【微批次】重新 Tokenize 的。仅根据一个微批次内的最长seq来进行填充
        """
        old_log_prob_microbatch = (
            old_log_probs[i:i + microbatch_size, :log_prob_microbatch.shape[-1]].to(device=model.device, dtype=torch.float32)
            if old_log_probs is not None else None
        ) # 需要根据 log_prob_microbatch 的维度对 old_log_prob_microbatch 进行截断
        raw_rewards_or_advantages_microbatch = raw_rewards_or_advantages[i:i + microbatch_size].unsqueeze(-1)
        per_token_policy_gradient_loss_microbatch, _ = run_compute_policy_gradient_loss(
            raw_rewards_or_advantages_microbatch,
            log_prob_microbatch,
            importance_reweighting_method,
            old_log_prob_microbatch,
            cliprange,
            bpe["response_mask"],
        )
        loss_microbatch = run_aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss_microbatch,
            bpe["response_mask"],
            loss_normalization,
            normalization_constant,
        )
        if loss_normalization == "sequence":
            # 序列归一化模式：按当前微批次大小占总 Batch 的比例缩放
            scale = len(inputs_microbatch) / len(repeated_prompts)
            loss_microbatch *= scale

        # 将本次的loss反向传播记录在optimizer中
        loss_microbatch.backward()
        loss += loss_microbatch
    # optimizer记录结束后要进行参数更新，.step()操作
    if max_grad_norm is not None:
        # 梯度截断
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    else:
        grad_norm_val = 0.0

    # 2. 优化器更新参数并清空梯度
    optimizer.step()
    optimizer.zero_grad()

    combined_metadata = {
        "grad_norm": grad_norm_val,
        **reward_metadata,
        **adv_metadata,
        **loss_metadata,
    }

    return loss, combined_metadata
    # raise NotImplementedError


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""
import json
import math
class my_Dataset(Dataset):
    # 注意时将整个格式化好的文档混合在一起进行分词，而不是每个问题回复对都分词
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle):
        docs = [] # 保存行数据列表
        template = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:\n{response}"
        )

        with open(dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    docs.append(template.format(instruction=data["prompt"], response=data["response"]))

        if shuffle:
            random.shuffle(docs)

        self.data = []
        all_ids = []
        # print(docs[0])
        for doc in docs:
            ids = tokenizer.encode(doc, add_special_tokens=False)
            # BOS/EOS每个行结尾都要有
            all_ids.extend([tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id])

        total_len = len(all_ids) - 1
        input_id = all_ids[: total_len]
        label = all_ids[1: total_len + 1]
        count = math.floor(total_len / seq_length)
        for i in range(count):
            item = {"input_ids": torch.tensor(input_id[i * seq_length: (i + 1) * seq_length], dtype=torch.long),
                    "labels": torch.tensor(label[i * seq_length: (i + 1) * seq_length], dtype=torch.long)}
            self.data.append(item)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    return my_Dataset(tokenizer, dataset_path, seq_length, shuffle)
    # raise NotImplementedError


class iterater_batches:
    def __init__(self, dataset, batch_size, shuffle):
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle

    # 为了适配外界的len()调用，只能将函数设置成__len__
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    # 为了适配外界的enumrate()调用，只能将函数设置成__iter__迭代函数
    def __iter__(self):
        # 生成 0 到 len(dataset)-1 的数字索引，不破坏原 dataset 结构
        # 必须用数字索引进行乱序及查找，因为自己实现的dataset类就是这样的，用index进行查找
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        length = len(indices)
        for i in range(0, length, self.batch_size):
            # 取出当前 batch 的数字索引列表，例如 [0, 1, 2, 3]
            batch_indices = indices[i: i + self.batch_size]

            # 根据数字索引 j 从 dataset[j] 中提取样本
            batch_input_ids = torch.stack(
                [self.dataset[j]["input_ids"] for j in batch_indices]
            )
            batch_labels = torch.stack(
                [self.dataset[j]["labels"] for j in batch_indices]
            )

            # yield 产出当前 batch
            yield {"input_ids": batch_input_ids, "labels": batch_labels}

def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return iterater_batches(dataset, batch_size, shuffle)
    # raise NotImplementedError

# 下面两个函数均是用于处理mmlu与gsm8k模型回复，既从模型回复中提炼出来最终的答案
# 虽然不用导入模型即可测试但需要训练模型进行配合，故未实现
def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    # 注意题目要求的形式!!!
    """
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    ### Instruction:
    XXX
    ### Response:
    XXX
    """
    prompt = (f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
              f"### Instruction:\n{prompt}\n\n")
    response_chosen = (
        f"### Response:\n{response_chosen}{tokenizer.eos_token}"
    )
    response_rejected = (
        f"### Response:\n{response_rejected}{tokenizer.eos_token}"
    )
    # 分词并转成词表 id
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    chosen_ids = tokenizer.encode(
        response_chosen, add_special_tokens=False
    )
    rejected_ids = tokenizer.encode(
        response_rejected, add_special_tokens=False
    )
    total_chosen_ids = prompt_ids + chosen_ids
    total_rejected_ids = prompt_ids + rejected_ids

    device = next(lm.parameters()).device
    total_chosen_ids_tensor = torch.tensor([total_chosen_ids], device=device)
    total_rejected_ids_tensor = torch.tensor([total_rejected_ids], device=device)

    # 制造掩码, 用于掩盖prompt
    chosen_mask = torch.tensor(
        [[0] * (len(prompt_ids) - 1) + [1] * len(chosen_ids)],
        dtype=torch.float32,
        device=device,
    )
    rejected_mask = torch.tensor(
        [[0] * (len(prompt_ids) - 1) + [1] * len(rejected_ids)],
        dtype=torch.float32,
        device=device,
    )

    # 计算选择文本 (Chosen) 的对数概率
    chosen_ids_lm_vocab = lm(total_chosen_ids_tensor[:, :-1]).logits
    chosen_ids_lm_ref_vocab = lm_ref(total_chosen_ids_tensor[:, :-1]).logits
    chosen_ids_lm_vocab_log_prob = F.log_softmax(chosen_ids_lm_vocab, dim=-1)
    chosen_ids_lm_ref_vocab_log_prob = F.log_softmax(chosen_ids_lm_ref_vocab, dim=-1)

    # 错位目标 tokens，准备 gather 索引，形状: [1, seq_len-1, 1]
    chosen_targets = total_chosen_ids_tensor[:, 1:].unsqueeze(-1)
    chosen_ids_lm_log = torch.gather(
        chosen_ids_lm_vocab_log_prob, -1, chosen_targets
    ).squeeze(-1)
    chosen_ids_lm_ref_log = torch.gather(
        chosen_ids_lm_ref_vocab_log_prob, -1, chosen_targets
    ).squeeze(-1)

    # 应用 mask 屏蔽 prompt 并沿序列维度求和 (.sum())，得到标量奖励值
    chosen_reward = ((chosen_ids_lm_log - chosen_ids_lm_ref_log) * chosen_mask).sum(dim=-1)

    # 4. 计算拒绝文本 (Rejected) 的对数概率
    rejected_ids_lm_vocab = lm(total_rejected_ids_tensor[:, :-1]).logits
    rejected_ids_lm_ref_vocab = lm_ref(total_rejected_ids_tensor[:, :-1]).logits
    rejected_ids_lm_vocab_log_prob = F.log_softmax(rejected_ids_lm_vocab, dim=-1)
    rejected_ids_lm_ref_vocab_log_prob = F.log_softmax(rejected_ids_lm_ref_vocab, dim=-1)

    rejected_targets = total_rejected_ids_tensor[:, 1:].unsqueeze(-1)
    rejected_ids_lm_log = torch.gather(
        rejected_ids_lm_vocab_log_prob, -1, rejected_targets
    ).squeeze(-1)
    rejected_ids_lm_ref_log = torch.gather(
        rejected_ids_lm_ref_vocab_log_prob, -1, rejected_targets
    ).squeeze(-1)

    rejected_reward = ((rejected_ids_lm_log - rejected_ids_lm_ref_log) * rejected_mask).sum(dim=-1)

    # 5. 计算 DPO Loss: -log(sigmoid(beta * (chosen_reward - rejected_reward)))
    logits = beta * (chosen_reward - rejected_reward)
    loss = -F.logsigmoid(logits).mean()

    return loss
    # raise NotImplementedError
