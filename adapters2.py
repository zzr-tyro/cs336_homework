from __future__ import annotations

import math
from typing import Any
import triton
import triton.language as tl
import torch
import torch.nn as nn

class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal): #is_casual是是否启动掩码，需支持
        B, N, d = Q.shape
        Br, Bc = 32, 32
        scale = 1.0 / math.sqrt(d)

        # 初始化全局输出 O, 最大值 m, 归一化项 l
        O = torch.zeros_like(Q)
        m = torch.full((B, N, 1), -float('inf'), device=Q.device)
        l = torch.zeros((B, N, 1), device=Q.device)

        # 1. 外层循环：遍历 Q 的块 (按 Br 步长推进)
        for j in range(0, N, Bc):
            # 切片选取当前 K 和 V 的区域 [B, H, j:j+Bc, d]
            K_j = K[:, j: j + Bc, :]
            V_j = V[:, j: j + Bc, :]

            # 2. 内层循环：遍历 K, V 的块 (按 Bc 步长推进)
            for i in range(0, N, Br):
                # 用切片选取当前 Q 的区域 [B, H, i:i+Br, d]
                Q_i = Q[:, i: i + Br, :]
                m_i = m[:, i: i + Br, :]
                l_i = l[:, i: i + Br, :]
                o_i = O[:, i: i + Br, :]

                # 3. 计算注意力得分矩阵 S_ij
                # Q_i: [B, H, cur_Br, d], K_j^T: [B, H, d, cur_Bc]
                # 算出的 S_ij 形状为 [B, H, cur_Br, cur_Bc]
                if not is_causal:
                    s_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
                    m_i_now = s_ij.max(dim=-1, keepdim=True).values
                    p_ij = torch.exp(s_ij - m_i_now)
                    l_i_now = p_ij.sum(dim=-1, keepdim=True)
                    m_i_max = torch.maximum(m_i, m_i_now)
                    exp1 = torch.exp(m_i - m_i_max)
                    exp2 = torch.exp(m_i_now - m_i_max)
                    l_i_new = l_i * exp1 + l_i_now * exp2

                else:
                    if i < j:
                        continue
                    elif i > j:
                        s_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
                    else:
                        s_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
                        mask = torch.tril(torch.ones(Br, Bc)).bool()
                        s_ij = s_ij.masked_fill(mask == 0, float("-inf"))

                    m_i_now = s_ij.max(dim=-1, keepdim=True).values
                    p_ij = torch.exp(s_ij - m_i_now)
                    l_i_now = p_ij.sum(dim=-1, keepdim=True)
                    m_i_max = torch.maximum(m_i, m_i_now)
                    exp1 = torch.exp(m_i - m_i_max)
                    exp2 = torch.exp(m_i_now - m_i_max)
                    l_i_new = l_i * exp1 + l_i_now * exp2

                #计算输出矩阵Oi
                o_i = 1.0 / l_i_new * (l_i * exp1 * o_i + exp2 * (p_ij @ V_j))

                # 放入存储器当中
                # 更新当前 Q block 的状态
                m[:, i:i + Br, :] = m_i_max
                l[:, i:i + Br, :] = l_i_new
                O[:, i:i + Br, :] = o_i

        lse = m.squeeze(-1) + torch.log(l.squeeze(-1))
        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        q_grad = torch.zeros_like(ctx.saved_tensors[0])
        k_grad = torch.zeros_like(ctx.saved_tensors[1])
        v_grad = torch.zeros_like(ctx.saved_tensors[2])

        B, N, d = ctx.saved_tensors[0].shape
        Br, Bc = 32, 32
        scale = 1.0 / math.sqrt(d)

        # 1. 外层循环：遍历 Q 的块 (按 Br 步长推进)
        for j in range(0, N, Bc):
            # 切片选取当前 K 和 V 的区域 [B, H, j:j+Bc, d]
            k_j = ctx.saved_tensors[1][:, j: j + Bc, :]
            v_j = ctx.saved_tensors[2][:, j: j + Bc, :]

            # 对保存梯度的矩阵进行切片操作用于保存计算的梯度
            v_grad_j = v_grad[:, j: j + Bc, :]
            k_grad_j = k_grad[:, j: j + Bc, :]

            # 2. 内层循环：遍历 K, V 的块 (按 Bc 步长推进)
            for i in range(0, N, Br):
                # 用切片选取当前 Q 的区域 [B, H, i:i+Br, d]
                q_i = ctx.saved_tensors[0][:, i: i + Br, :]
                o_i = ctx.saved_tensors[3][:, i: i + Br, :]
                lse_i = ctx.saved_tensors[4][:, i: i + Br].unsqueeze(-1)
                grad_o_i = grad_outputs[0][:, i: i + Br, :]

                # 对保存梯度的矩阵进行切片操作用于保存计算的梯度
                q_grad_i = q_grad[:, i: i + Br, :]

                if not ctx.is_causal:
                    # 计算注意力得分矩阵 S_ij
                    # Q_i: [B, H, cur_Br, d], K_j^T: [B, H, d, cur_Bc], 算出的 S_ij 形状为 [B, H, cur_Br, cur_Bc]
                    s_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) * scale
                    p_ij = torch.exp(s_ij - lse_i)

                    #计算v矩阵的梯度
                    v_grad_j = v_grad_j + p_ij.transpose(-2, -1) @ grad_o_i

                    #计算p矩阵梯度
                    p_ij_grad = grad_o_i @ v_j.transpose(-2, -1)

                    #计算s矩阵的梯度
                    d_i = torch.sum(o_i * grad_o_i, dim = -1, keepdim = True)
                    s_ij_grad = p_ij * (p_ij_grad - d_i)

                    #计算Q,K矩阵的梯度
                    q_grad_i = q_grad_i + s_ij_grad @ k_j * scale
                    k_grad_j = k_grad_j + s_ij_grad.transpose(-2, -1) @ q_i * scale

                else:
                    if i > j:
                        continue
                    elif j < i:
                        # 计算注意力得分矩阵 S_ij
                        s_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) * scale
                    else:
                        s_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) * scale
                        mask = torch.tril(torch.ones(Br, Bc)).bool()
                        s_ij = s_ij.masked_fill(mask == 0, float("-inf"))

                    p_ij = torch.exp(s_ij - lse_i)
                    # 计算v矩阵的梯度
                    v_grad_j = v_grad_j + p_ij.transpose(-2, -1) @ grad_o_i
                    # 计算p矩阵梯度
                    p_ij_grad = grad_o_i @ v_j.transpose(-2, -1)
                    # 计算s矩阵的梯度
                    d_i = torch.sum(o_i * grad_o_i, dim=-1, keepdim=True)
                    s_ij_grad = p_ij * (p_ij_grad - d_i)
                    # 计算Q,K矩阵的梯度
                    q_grad_i = q_grad_i + s_ij_grad @ k_j * scale
                    k_grad_j = k_grad_j + s_ij_grad.transpose(-2, -1) @ q_i * scale

                v_grad[:, j: j + Bc, :] = v_grad_j
                q_grad[:, i: i + Br, :] = q_grad_i
                k_grad[:, j: j + Bc, :] = k_grad_j

        return q_grad, k_grad, v_grad, None


def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass

    return MyFlashAttnAutogradFunctionClass
    raise NotImplementedError

@triton.jit
def flash_fwd_kernel(Q_ptr, K_ptr, V_ptr,
    O_ptr, lse_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lseb, stride_lseq,
    N_QUERIES, N_KEYS,
    scale, # 1/sqrt(d)
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, # Br
    K_TILE_SIZE: tl.constexpr, # Bc
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0) # grid传入的Q矩阵分开序号在这里传入
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor

    # 定义分块矩阵的范式
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    lse_block_ptr = tl.make_block_ptr(
        lse_ptr + batch_index * stride_lseb,
        shape=(N_QUERIES,),
        strides=(stride_lseq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # 初始化片上缓冲区 (片上累加器必须为 float32 类型)
    o_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m_i = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)

    q_i = tl.load(Q_block_ptr)

    if not is_causal:
        for j in range(0, N_KEYS, K_TILE_SIZE):
            k_j = tl.load(K_block_ptr)
            v_j = tl.load(V_block_ptr)
            s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
            m_i_now = tl.max(s_ij, axis = -1)
            m_i_max = tl.maximum(m_i, m_i_now)
            p_ij = tl.exp(s_ij - m_i_max[:, None])
            l_i_now = tl.sum(p_ij, axis = -1)
            exp1 = tl.exp(m_i - m_i_max)
            m_i = m_i_max
            l_i = l_i * exp1 + l_i_now
            # 计算输出矩阵Oi
            #p_ij = p_ij.to(v_j.dtype)  # 显式强转为 fp16/bf16
            o_i = (exp1[:, None] * o_i + tl.dot(p_ij, v_j)) # 可能存在类型不匹配的问题

            # 向后移动分块
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    else:
        #triton编译器不支持continue关键字
        for j in range(0, N_KEYS, K_TILE_SIZE):
            if query_tile_index > j // K_TILE_SIZE:
                k_j = tl.load(K_block_ptr)
                v_j = tl.load(V_block_ptr)
                s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
                m_i_now = tl.max(s_ij, axis=-1)
                m_i_max = tl.maximum(m_i, m_i_now)
                p_ij = tl.exp(s_ij - m_i_max[:, None])
                l_i_now = tl.sum(p_ij, axis=-1)
                exp1 = tl.exp(m_i - m_i_max)
                m_i = m_i_max
                l_i = l_i * exp1 + l_i_now
                # 计算输出矩阵Oi
                #p_ij = p_ij.to(v_j.dtype)  # 显式强转为 fp16/bf16
                o_i = (exp1[:, None] * o_i + tl.dot(p_ij, v_j))  # 可能存在类型不匹配的问题
            elif query_tile_index == j // K_TILE_SIZE:
                k_j = tl.load(K_block_ptr)
                v_j = tl.load(V_block_ptr)
                s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
                offs_m = tl.arange(0, Q_TILE_SIZE)
                offs_n = tl.arange(0, K_TILE_SIZE)
                mask = offs_m[:, None] < offs_n[None, :]
                s_ij = tl.where(mask, float("-inf"), s_ij)
                m_i_now = tl.max(s_ij, axis=-1)
                m_i_max = tl.maximum(m_i, m_i_now)
                p_ij = tl.exp(s_ij - m_i_max[:, None])
                l_i_now = tl.sum(p_ij, axis=-1)
                exp1 = tl.exp(m_i - m_i_max)
                m_i = m_i_max
                l_i = l_i * exp1 + l_i_now
                # 计算输出矩阵Oi
                #p_ij = p_ij.to(v_j.dtype)  # 显式强转为 fp16/bf16
                o_i = (exp1[:, None] * o_i + tl.dot(p_ij, v_j))  # 可能存在类型不匹配的问题

            # 向后移动分块
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))


    # 在循环外归一化o_i，在内部不用进行乘l_i的操作进行还原
    o_i = o_i / l_i[:, None]

    # 计算 Log-Sum-Exp（lse）
    lse_i = m_i + tl.log(l_i)

    # 保存输出矩阵 O 与 lse
    tl.store(O_block_ptr, o_i.to(O_ptr.type.element_ty))
    tl.store(lse_block_ptr, lse_i.to(lse_ptr.type.element_ty))

@triton.jit
def flash_bwd_kernel_q(Q_ptr, K_ptr, V_ptr, O_ptr,
    Q_grad_ptr, O_grad_ptr, lse_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_qgb, stride_qgq, stride_qgd,
    stride_ogb, stride_ogq, stride_ogd,
    stride_lseb, stride_lseq,
    N_QUERIES, N_KEYS,
    scale, # 1/sqrt(d)
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, # Br
    K_TILE_SIZE: tl.constexpr, # Bc
    is_causal: tl.constexpr
):
    # grid传入的Q矩阵分开序号在这里传入
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 定义分块矩阵的范式
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Ograd_block_ptr = tl.make_block_ptr(
        O_grad_ptr + batch_index * stride_ogb,
        shape=(N_QUERIES, D),
        strides=(stride_ogq, stride_ogd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Qgrad_block_ptr = tl.make_block_ptr(
        Q_grad_ptr + batch_index * stride_qgb,
        shape=(N_QUERIES, D),
        strides=(stride_qgq, stride_qgd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    lse_block_ptr = tl.make_block_ptr(
        lse_ptr + batch_index * stride_lseb,
        shape=(N_QUERIES,),
        strides=(stride_lseq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # 初始化片上缓冲区 (片上累加器必须为 float32 类型)
    q_grad_i = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    q_i = tl.load(Q_block_ptr)
    o_i = tl.load(O_block_ptr)
    o_grad_i = tl.load(Ograd_block_ptr)
    lse_i = tl.load(lse_block_ptr)

    if not is_causal:
        for j in range(0, N_KEYS, K_TILE_SIZE):
            k_j = tl.load(K_block_ptr)
            v_j = tl.load(V_block_ptr)
            s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
            p_ij = tl.exp(s_ij - lse_i[:, None])
            # 计算p矩阵梯度
            p_ij_grad = tl.dot(o_grad_i, tl.trans(v_j))
            # 计算s矩阵的梯度
            d_i = tl.sum(o_i * o_grad_i, axis=-1)
            s_ij_grad = p_ij * (p_ij_grad - d_i[:, None])
            # 计算Q,K矩阵的梯度
            q_grad_i = q_grad_i + tl.dot(s_ij_grad, k_j) * scale

            # 向后移动分块
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    else:
        for j in range(0, N_KEYS, K_TILE_SIZE):
            if query_tile_index > j // K_TILE_SIZE:
                k_j = tl.load(K_block_ptr)
                v_j = tl.load(V_block_ptr)
                s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
                p_ij = tl.exp(s_ij - lse_i[:, None])
                # 计算p矩阵梯度
                p_ij_grad = tl.dot(o_grad_i, tl.trans(v_j))
                # 计算s矩阵的梯度
                d_i = tl.sum(o_i * o_grad_i, axis=-1)
                s_ij_grad = p_ij * (p_ij_grad - d_i[:, None])
                # 计算Q,K矩阵的梯度
                q_grad_i = q_grad_i + tl.dot(s_ij_grad, k_j) * scale

                # 向后移动分块
                K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
                V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

            elif j // K_TILE_SIZE == query_tile_index:
                k_j = tl.load(K_block_ptr)
                v_j = tl.load(V_block_ptr)
                s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
                offs_m = tl.arange(0, Q_TILE_SIZE)
                offs_n = tl.arange(0, K_TILE_SIZE)
                mask = offs_m[:, None] < offs_n[None, :]
                s_ij = tl.where(mask, float("-inf"), s_ij)
                p_ij = tl.exp(s_ij - lse_i[:, None])
                # 计算p矩阵梯度
                p_ij_grad = tl.dot(o_grad_i, tl.trans(v_j))
                # 计算s矩阵的梯度
                d_i = tl.sum(o_i * o_grad_i, axis=-1)
                s_ij_grad = p_ij * (p_ij_grad - d_i[:, None])
                # 计算Q,K矩阵的梯度
                q_grad_i = q_grad_i + tl.dot(s_ij_grad, k_j) * scale

                # 向后移动分块
                K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
                V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    tl.store(Qgrad_block_ptr, q_grad_i.to(Qgrad_block_ptr.type.element_ty))


@triton.jit
def flash_bwd_kernel_kv(Q_ptr, K_ptr, V_ptr, O_ptr, # 要注意固定什么分块矩阵（offset要偏移），取什么分块矩阵（要在循环中advance，offset为0）
        K_grad_ptr, V_grad_ptr, O_grad_ptr, lse_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_kgb, stride_kgq, stride_kgd,
        stride_vgb, stride_vgq, stride_vgd,
        stride_ogb, stride_ogq, stride_ogd,
        stride_lseb, stride_lseq,
        N_QUERIES, N_KEYS,
        scale,  # 1/sqrt(d)
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,  # Br
        K_TILE_SIZE: tl.constexpr,  # Bc
        is_causal: tl.constexpr
):
    # grid传入的Q矩阵分开序号在这里传入
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 定义分块矩阵的范式
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Ograd_block_ptr = tl.make_block_ptr(
        O_grad_ptr + batch_index * stride_ogb,
        shape=(N_QUERIES, D),
        strides=(stride_ogq, stride_ogd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Kgrad_block_ptr = tl.make_block_ptr(
        K_grad_ptr + batch_index * stride_kgb,
        shape=(N_KEYS, D),
        strides=(stride_kgq, stride_kgd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Vgrad_block_ptr = tl.make_block_ptr(
        V_grad_ptr + batch_index * stride_vgb,
        shape=(N_KEYS, D),
        strides=(stride_vgq, stride_vgd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    lse_block_ptr = tl.make_block_ptr(
        lse_ptr + batch_index * stride_lseb,
        shape=(N_QUERIES,),
        strides=(stride_lseq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # 初始化片上缓冲区 (片上累加器必须为 float32 类型)
    k_grad_j = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    v_grad_j = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)

    k_j = tl.load(K_block_ptr)
    v_j = tl.load(V_block_ptr)


    if not is_causal:
        for i in range(0, N_QUERIES, Q_TILE_SIZE):
            q_i = tl.load(Q_block_ptr)
            o_i = tl.load(O_block_ptr)
            o_grad_i = tl.load(Ograd_block_ptr)
            lse_i = tl.load(lse_block_ptr)
            s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
            p_ij = tl.exp(s_ij - lse_i[:, None])
            # 计算v矩阵的梯度
            v_grad_j = v_grad_j + tl.dot(tl.trans(p_ij), o_grad_i)
            # 计算p矩阵梯度
            p_ij_grad = tl.dot(o_grad_i, tl.trans(v_j))
            # 计算s矩阵的梯度
            d_i = tl.sum(o_i * o_grad_i, axis=-1)
            s_ij_grad = p_ij * (p_ij_grad - d_i[:, None])
            # 计算Q,K矩阵的梯度
            k_grad_j = k_grad_j + tl.dot(tl.trans(s_ij_grad), q_i) * scale

            # 向后移动分块
            Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
            O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
            Ograd_block_ptr = tl.advance(Ograd_block_ptr, (Q_TILE_SIZE, 0))
            lse_block_ptr = tl.advance(lse_block_ptr, (Q_TILE_SIZE, ))

    else:
        for i in range(0, N_QUERIES, Q_TILE_SIZE):
            if key_tile_index < i // Q_TILE_SIZE:
                q_i = tl.load(Q_block_ptr)
                o_i = tl.load(O_block_ptr)
                o_grad_i = tl.load(Ograd_block_ptr)
                lse_i = tl.load(lse_block_ptr)
                s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
                p_ij = tl.exp(s_ij - lse_i[:, None])
                # 计算v矩阵的梯度
                v_grad_j = v_grad_j + tl.dot(tl.trans(p_ij), o_grad_i)
                # 计算p矩阵梯度
                p_ij_grad = tl.dot(o_grad_i, tl.trans(v_j))
                # 计算s矩阵的梯度
                d_i = tl.sum(o_i * o_grad_i, axis=-1)
                s_ij_grad = p_ij * (p_ij_grad - d_i[:, None])
                # 计算Q,K矩阵的梯度
                k_grad_j = k_grad_j + tl.dot(tl.trans(s_ij_grad), q_i) * scale



            elif key_tile_index == i // Q_TILE_SIZE:
                q_i = tl.load(Q_block_ptr)
                o_i = tl.load(O_block_ptr)
                o_grad_i = tl.load(Ograd_block_ptr)
                lse_i = tl.load(lse_block_ptr)
                s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
                offs_m = tl.arange(0, Q_TILE_SIZE)
                offs_n = tl.arange(0, K_TILE_SIZE)
                mask = offs_m[:, None] < offs_n[None, :]
                s_ij = tl.where(mask, float("-inf"), s_ij)
                p_ij = tl.exp(s_ij - lse_i[:, None])
                # 计算v矩阵的梯度
                v_grad_j = v_grad_j + tl.dot(tl.trans(p_ij), o_grad_i)
                # 计算p矩阵梯度
                p_ij_grad = tl.dot(o_grad_i, tl.trans(v_j))
                # 计算s矩阵的梯度
                d_i = tl.sum(o_i * o_grad_i, axis=-1)
                s_ij_grad = p_ij * (p_ij_grad - d_i[:, None])
                # 计算Q,K矩阵的梯度
                k_grad_j = k_grad_j + tl.dot(tl.trans(s_ij_grad), q_i) * scale

            # 向后移动分块
            Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
            O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
            Ograd_block_ptr = tl.advance(Ograd_block_ptr, (Q_TILE_SIZE, 0))
            lse_block_ptr = tl.advance(lse_block_ptr, (Q_TILE_SIZE,))


    tl.store(Kgrad_block_ptr, k_grad_j.to(Kgrad_block_ptr.type.element_ty))
    tl.store(Vgrad_block_ptr, v_grad_j.to(Vgrad_block_ptr.type.element_ty))

class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        batch_size = Q.shape[0]
        n_queries = Q.shape[1]
        n_keys = K.shape[1]
        d = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        O = torch.empty_like(Q)
        lse = torch.empty((batch_size, n_queries), device=Q.device, dtype=torch.float32)

        Q_tile_size = 32
        K_tile_size = 32

        grid = (triton.cdiv(n_queries, Q_tile_size), batch_size)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, lse,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            lse.stride(0), lse.stride(1),
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=scale,
            D=d,
            Q_TILE_SIZE=Q_tile_size,
            K_TILE_SIZE=K_tile_size,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, lse)
        ctx.is_causal = is_causal

        return  O

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        # 提取ctx中存入的矩阵
        Q = ctx.saved_tensors[0]
        K = ctx.saved_tensors[1]
        V = ctx.saved_tensors[2]
        O = ctx.saved_tensors[3]
        lse = ctx.saved_tensors[4]

        # 提取并设置必要的参数
        batch_size = Q.shape[0]
        n_queries = Q.shape[1]
        n_keys = K.shape[1]
        d = Q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        Q_tile_size = 32
        K_tile_size = 32

        # 初始化矩阵导数空间
        Q_grad = torch.empty_like(Q)
        K_grad = torch.empty_like(K)
        V_grad = torch.empty_like(V)

        #设置网格
        grid1 = (triton.cdiv(n_queries, Q_tile_size), batch_size)
        grid2 = (triton.cdiv(n_keys, K_tile_size), batch_size)

        flash_bwd_kernel_q[grid1](
            Q, K, V, O,
            Q_grad, grad_outputs[0], lse,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            Q_grad.stride(0), Q_grad.stride(1), Q_grad.stride(2),
            grad_outputs[0].stride(0), grad_outputs[0].stride(1), grad_outputs[0].stride(2),
            lse.stride(0), lse.stride(1),
            n_queries, n_keys,
            scale,  # 1/sqrt(d)
            d,
            Q_tile_size,  # Br
            K_tile_size,  # Bc
            ctx.is_causal
        )

        flash_bwd_kernel_kv[grid2](
            Q, K, V, O,
            K_grad, V_grad, grad_outputs[0], lse,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            K_grad.stride(0), K_grad.stride(1), K_grad.stride(2),
            V_grad.stride(0), V_grad.stride(1), V_grad.stride(2),
            grad_outputs[0].stride(0), grad_outputs[0].stride(1), grad_outputs[0].stride(2),
            lse.stride(0), lse.stride(1),
            n_queries, n_keys,
            scale,  # 1/sqrt(d)
            d,
            Q_tile_size,  # Br
            K_tile_size,  # Bc
            ctx.is_causal
        )

        return Q_grad, K_grad, V_grad, None


def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    return MyTritonFlashAttentionAutogradFunctionClass
    raise NotImplementedError

import torch.distributed as dist
class DDP1(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []  # 用于存储异步 all_reduce 的 work handle

        # 广播 Rank 0 的参数到所有其它 Rank
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()

        for param in self.module.parameters():
            if param.grad is not None and param.requires_grad:
                param.grad.data.div_(world_size)
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class DDP2(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []  # 用于存储异步 all_reduce 的 work handle

        # 1. 广播 Rank 0 的参数到所有其它 Rank
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    #这里改成将所有参数合并，进行规约操作
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()

        # 收集所有需要求梯度的参数
        params_with_grad = [p for p in self.module.parameters() if p.requires_grad and p.grad is not None]
        if not params_with_grad:
            return

        # 1. 把所有参数的梯度 Flatten 并拼接（Cat）成一个大一维 Tensor
        flat_grads = torch.cat([p.grad.view(-1) for p in params_with_grad])

        # 2. 先除以 world_size 求平均，然后发起单次 Batched All-Reduce 通信
        flat_grads.div_(world_size)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)

        # 3. 将通信完的大 Tensor 切片拆解，重新写回各个 param.grad
        offset = 0
        for p in params_with_grad:
            numel = p.grad.numel()
            p.grad.copy_(flat_grads[offset: offset + numel].view_as(p.grad))
            offset += numel

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class DDP3(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []  # 用于存储异步 all_reduce 的 work handle

        # 1. 广播 Rank 0 的参数到所有其它 Rank
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 2. 为每个需要梯度的参数注册 Hook，实现 Overlap
        self._register_hooks()

    # 因为分布式计算分属不同的进程，所以需要将规约的参数进行自动处理。由于不清楚进程完成的细节，所以需要将parma绑定到hookfn上，用于自动进行规约
    def _register_hooks(self):
        world_size = dist.get_world_size()

        for param in self.module.parameters():
            if param.requires_grad:
                def hook_fn(p):
                    if p.grad is not None:
                        # 先除以 world_size 求平均
                        p.grad.data.div_(world_size)
                        # 异步发起 All-Reduce (SUM)
                        handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
                        self.handles.append(handle)

                # 使用 PyTorch 推荐的 post accumulate grad hook
                param.register_post_accumulate_grad_hook(hook_fn)

    def finish_gradient_synchronization(self):
        """等待所有梯度的异步传输完成"""
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

# 用于创建类
def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDP(module)
    # DDP1显示 48.71 DDP2显示38.00 DDP3显示 40.15
    return DDP3(module)
    raise NotImplementedError

#这个是用于执行类中规约函数的，其运行位置如其名ddp_on_after_backward
def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    ddp_model.finish_gradient_synchronization()
    # raise NotImplementedError


from functools import partial
from cs336_basics.model import Embedding, Linear, RMSNorm

# 1. 存储分片参数元数据的结构体
class ParamInfo:
    def __init__(self, block_name, initial_shape, world_size, rank, size, padding, param, module):
        self.name = block_name
        self.shape = initial_shape
        self.world_size = world_size
        self.rank = rank
        self.size = size
        self.padding = padding
        self.param = param  # 当前 rank 对应的分片 nn.Parameter
        self.module = module  # 子模块引用 (nn.Linear 或 nn.Embedding)

# 由于autograd的限制必须自己实现要进行分片的层，如果使用勾函数则会出现将注册的参数释放导致无法反传的情况
# 2. 线性层 (Linear) 自定义 Autograd 节点
class FSDPLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard_param, padding, orig_shape, compute_dtype, world_size):
        # 1. 通信准备与 dtype 转换
        param_data = shard_param.data
        if compute_dtype is not None:
            param_data = param_data.to(compute_dtype)

        # 2. 执行 All-Gather 拼接全量权重 full_weight
        # 设置存储空间
        gathered_list = [torch.empty_like(param_data) for _ in range(world_size)]
        # 通信传送参数
        dist.all_gather(gathered_list, param_data)

        full_weight_flat = torch.cat(gathered_list, dim=0)
        if padding > 0:
            full_weight_flat = full_weight_flat[:-padding]
        full_weight = full_weight_flat.reshape(orig_shape)

        # 3. 前向计算矩阵乘法
        x_calc = x.to(compute_dtype) if compute_dtype is not None else x
        full_weight = full_weight.to(compute_dtype) if compute_dtype is not None else full_weight

        # out = x @ W^T
        out = torch.einsum("...i, oi -> ...o", x_calc, full_weight)

        # 4. 上下文保存：【只保存 x，绝对不保存 full_weight，计算完立即释放内存】
        ctx.save_for_backward(x)
        ctx.shard_param = shard_param
        ctx.padding = padding
        ctx.orig_shape = orig_shape
        ctx.compute_dtype = compute_dtype
        ctx.world_size = world_size

        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        shard_param = ctx.shard_param
        world_size = ctx.world_size

        # 1. 反向传播阶段：重新 All-Gather 收集 full_weight
        param_data = shard_param.data
        if ctx.compute_dtype is not None:
            param_data = param_data.to(ctx.compute_dtype)

        gathered_list = [torch.empty_like(param_data) for _ in range(world_size)]
        dist.all_gather(gathered_list, param_data)

        full_weight_flat = torch.cat(gathered_list, dim=0)
        if ctx.padding > 0:
            full_weight_flat = full_weight_flat[:-ctx.padding]
        full_weight = full_weight_flat.reshape(ctx.orig_shape)

        # 转换 dtype 用于计算
        grad_out_calc = grad_output.to(ctx.compute_dtype) if ctx.compute_dtype is not None else grad_output
        full_weight = full_weight.to(ctx.compute_dtype) if ctx.compute_dtype is not None else full_weight
        x_calc = x.to(ctx.compute_dtype) if ctx.compute_dtype is not None else x

        # 2. 手动链式法则求导
        # (1) grad_x = grad_output @ full_weight
        grad_x = torch.einsum("...o, oi -> ...i", grad_out_calc, full_weight)

        # (2) grad_full_weight = grad_output^T @ x
        # 将前两个维度batch_size与sequence_length展平
        grad_out_flat = grad_out_calc.reshape(-1, grad_out_calc.shape[-1])
        x_flat = x_calc.reshape(-1, x_calc.shape[-1])
        grad_full_weight = grad_out_flat.T @ x_flat  # Shape: [d_out, d_in]

        # 3. 对权重的全量梯度执行 Reduce-Scatter， 展平并填充
        grad_flat = grad_full_weight.flatten()
        if ctx.padding > 0:
            grad_flat = torch.cat([
                grad_flat,
                torch.zeros(ctx.padding, dtype=grad_flat.dtype, device=grad_flat.device)
            ])

        chunk_size = grad_flat.numel() // world_size
        # 创建缓存池
        shard_grad_buffer = torch.empty(chunk_size, dtype=grad_flat.dtype, device=grad_flat.device)
        dist.reduce_scatter_tensor(shard_grad_buffer, grad_flat, op=dist.ReduceOp.SUM)

        # 4. 累加梯度给当前 rank 的主权重 shard_param.grad,把其他rank的梯度累加
        # 除以batch_size的活在计算loss的时候就已经干了,但是还需除以world_size
        shard_grad_fp32 = shard_grad_buffer.to(torch.float32)
        shard_grad_fp32 /= world_size
        if shard_param.grad is None:
            shard_param.grad = shard_grad_fp32
        else:
            shard_param.grad += shard_grad_fp32

        # 5. 返回值与 forward 的输入形参按顺序逐一对应：
        # forward(ctx, x, shard_param, padding, orig_shape, compute_dtype, world_size)
        return grad_x.to(x.dtype), None, None, None, None, None


# 3. 词表层 (Embedding) 自定义 Autograd 节点
class FSDPEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard_param, padding, orig_shape, compute_dtype, world_size):
        # 1. 拼接全量词表权重
        param_data = shard_param.data
        if compute_dtype is not None:
            param_data = param_data.to(compute_dtype)

        gathered_list = [torch.empty_like(param_data) for _ in range(world_size)]
        dist.all_gather(gathered_list, param_data)

        full_weight_flat = torch.cat(gathered_list, dim=0)
        if padding > 0:
            full_weight_flat = full_weight_flat[:-padding]
        full_weight = full_weight_flat.reshape(orig_shape)

        # 2. 查表计算
        out = nn.functional.embedding(x, full_weight)

        # 3. 保存上下文 (Token ID 输入 x 占用极小)
        ctx.save_for_backward(x)
        ctx.shard_param = shard_param
        ctx.padding = padding
        ctx.orig_shape = orig_shape
        ctx.world_size = world_size

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        shard_param = ctx.shard_param
        world_size = ctx.world_size

        # 1. 根据索引 x 稀疏累加算出全量 grad_full_weight
        grad_full_weight = torch.zeros(
            ctx.orig_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        # Embedding 前向传播本质上是一个“按索引查表（Lookup）”的过程，因此反向传播就是前向的逆过程 ——“按索引把梯度累加回权重矩阵”。
        # 根据tokenid加到与词向量矩阵一样大小的0矩阵上，如果有id相同就会累加
        grad_full_weight.index_add_(
            #在 PyTorch 的 reshape 中第一个参数 -1 代表“自动计算/推导这个维度的大小”。 它的值等于 Tensor 总元素个数 / 其他已知维度的乘积
            0, x.reshape(-1), grad_output.reshape(-1, grad_output.shape[-1])
        )

        # 2. 执行 Reduce-Scatter
        grad_flat = grad_full_weight.flatten()
        if ctx.padding > 0:
            grad_flat = torch.cat([
                grad_flat,
                torch.zeros(ctx.padding, dtype=grad_flat.dtype, device=grad_flat.device)
            ])

        chunk_size = grad_flat.numel() // world_size
        shard_grad_buffer = torch.empty(chunk_size, dtype=grad_flat.dtype, device=grad_flat.device)

        dist.reduce_scatter_tensor(shard_grad_buffer, grad_flat, op=dist.ReduceOp.SUM)

        # 3. 累加梯度给主权重
        shard_grad_fp32 = shard_grad_buffer.to(torch.float32)
        shard_grad_fp32 /= world_size
        if shard_param.grad is None:
            shard_param.grad = shard_grad_fp32
        else:
            shard_param.grad += shard_grad_fp32

        # 输入 token_id 不需要梯度，返回 None
        return None, None, None, None, None, None



# 4. 用于包裹子模块 forward 的替换函数
def custom_linear_forward(x, *args, p_info=None, compute_dtype=None, world_size=None, **kwargs):
    # apply 用于传入必要的参数，在forward自动一层一层执行的时候只会自动传入ctx
    return FSDPLinearFunction.apply(
        x,
        p_info.param,
        p_info.padding,
        p_info.shape,
        compute_dtype,
        world_size
    )


def custom_embedding_forward(x, *args, p_info=None, compute_dtype=None, world_size=None, **kwargs):
    return FSDPEmbeddingFunction.apply(
        x,
        p_info.param,
        p_info.padding,
        p_info.shape,
        compute_dtype,
        world_size
    )


# ==========================================
# 5. FSDP 核心 Wrapper 类
# ==========================================
class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.partial_params = []

        # 遍历原模块参数进行切片
        for name, param in list(module.named_parameters()):
            module_name, param_name = name.rsplit(".", 1)
            submodule = module.get_submodule(module_name)

            # 排除不需要切片的 Norm 层
            if isinstance(submodule, (nn.Linear, Linear, nn.Embedding, Embedding)):
                param_flatten = torch.flatten(param.data)
                total_len = param_flatten.numel()
                size = math.ceil(total_len / self.world_size)
                padding = size * self.world_size - total_len

                if self.rank == self.world_size - 1:
                    shared_param = torch.nn.Parameter(torch.cat([
                        param_flatten[size * self.rank:],
                        torch.zeros(padding, dtype=param_flatten.dtype, device=param_flatten.device)
                    ]))
                else:
                    shared_param = torch.nn.Parameter(param_flatten[size * self.rank: size * (self.rank + 1)])

                # 更新原 submodule 的参数指针
                setattr(submodule, param_name, shared_param)

                partial_param = ParamInfo(
                    module_name,
                    param.shape,
                    self.world_size,
                    self.rank,
                    size,
                    padding,
                    shared_param,
                    submodule
                )
                self.partial_params.append(partial_param)

                # 核心步骤：动态替换 (Monkey Patching) 子模块的 .forward 方法
                """
                当你在前向传播中运行 FSDPLinearFunction.apply(...) 时，PyTorch 底层会自动完成以下操作：
                自动创建计算图节点（Graph Node）：
                PyTorch 会在 C++ 计算图里生成一个属于 FSDPLinearFunction 的求导节点，并把你在 FSDPLinearFunction 类里写好的 staticmethod backward 关联到这个节点上。
                自动捕获反向传播路径：
                PyTorch 的 Autograd 引擎根本不关心子模块 submodule 本身有没有 backward 方法，它只关心：“这个 Output Tensor 是由哪一个 autograd.Function 的 apply 算出来的？”
                反向传播时的自动触发：
                当外部调用 loss.backward() 时，Autograd 引擎沿着计算图倒推，会自动调用你写在 FSDPLinearFunction 类里的那个 backward 静态方法。
                """
                if isinstance(submodule, (nn.Linear, Linear)):
                    submodule.forward = partial(
                        custom_linear_forward,
                        p_info=partial_param,
                        compute_dtype=self.compute_dtype,
                        world_size=self.world_size
                    )
                elif isinstance(submodule, (nn.Embedding, Embedding)):
                    submodule.forward = partial(
                        custom_embedding_forward,
                        p_info=partial_param,
                        compute_dtype=self.compute_dtype,
                        world_size=self.world_size
                    )

            else:
                # Norm 层的处理：不切片，保持全量 nn.Parameter，Autograd 自动求导
                full_param = ParamInfo(
                    module_name,
                    param.shape,
                    self.world_size,
                    self.rank,
                    param.numel(),
                    -1,
                    param,
                    submodule
                )
                self.partial_params.append(full_param)

    def forward(self, *args, **kwargs):
        # 内部 submodule 触发调用时，会自动落入被替换后的 custom_forward 逻辑
        return self.module(*args, **kwargs)

    # finish_gradient_synchronization 是专门用于在反向传播（Backward）完成后，统一完成/清理所有参数梯度同步的核心回调接口。
    # 如果有异步通信的情况需要统一等待
    # 此处仅需处理未进行划分的参数，划分的参数在自定义的类中已经处理好了
    def finish_gradient_synchronization(self):
        for p_info in self.partial_params:
            # 如果是未切片的全量参数（如 Norm 层，padding == -1 或 size == numel）
            if p_info.padding == -1 and p_info.param.grad is not None:
                # 对全量参数的梯度做 All-Reduce 平均
                dist.all_reduce(p_info.param.grad, op=dist.ReduceOp.SUM)
                p_info.param.grad /= self.world_size


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    fully-sharded data parallel training, including weight sharding,
    all-gather for forward/backward, and gradient reduce-scatter.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with FSDP.
        compute_dtype: optional torch.dtype
            If provided, weights are cast to this dtype before communication
            and compute, saving bandwidth. Master weights stay in fp32.
    Returns:
        Instance of an FSDP class.
    """
    # For example: return FSDP(module, compute_dtype=compute_dtype)
    return FSDP(module, compute_dtype=compute_dtype)
    # raise NotImplementedError


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the FSDP-wrapped model.
    """
    # For example: fsdp_model.finish_gradient_synchronization()
    fsdp_model.finish_gradient_synchronization()
    # raise NotImplementedError


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    All-gather sharded parameters from the FSDP model to reconstruct full
    parameter tensors. Replicated parameters are returned as-is.

    Args:
        fsdp_model: torch.nn.Module
            FSDP-wrapped model.
    Returns:
        State dictionary mapping parameter names to full (unsharded) tensors.
    """
    full_params = {}
    world_size = dist.get_world_size()
    # 直接遍历 FSDP 中保存的 partial_params 列表
    for p in fsdp_model.partial_params:
        # 参数的全路径名称（例如: "layers.0.feed_forward.w1.weight"）
        param_full_name = f"{p.name}.weight"
        if p.padding >= 0:
            # 准备接收容器并执行 All-Gather
            param_data = p.param.data
            gathered_list = [torch.empty_like(param_data) for _ in range(world_size)]
            dist.all_gather(gathered_list, param_data)

            # 拼接并还原为原始 Shape
            full_weight_flat = torch.cat(gathered_list, dim=0)
            if p.padding > 0:
                full_weight_flat = full_weight_flat[:-p.padding]

            full_weight = full_weight_flat.reshape(p.shape)

            # 存入字典（使用 .detach().clone() 防止计算图污染）
            full_params[param_full_name] = full_weight.detach().clone()

        # 2. 未切分的参数 (如 Norm 层，padding == -1)，直接复制
        else:
            full_params[param_full_name] = p.param.data.detach().clone()

    return full_params
    # raise NotImplementedError

# 获取节点编号，按照某种规则划分参数，根据划分的参数创建标准的adam，传入所需的部分梯度数据(每个Parameter本身就带着自己的.grad)
# 计算并更新参数，随后传播(all_scatter)更新后的参数
class optimizer_shared(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs): # params是一个迭代器，用一次遍历就失效了

        # 每个进程在运行get_rank() / get_world_size()函数时就可以知道其编号， 在创建时就存储下来了
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # 必须先生成params_shared然后用params_shared构造父类
        params_sharded = list(params)
        # 保存一下整个参数列表，为后面参数扩散做铺垫
        self.params = params_sharded
        p_size = len(params_sharded)
        self.p_size = p_size
        start = (p_size // self.world_size) * self.rank
        if self.rank == self.world_size - 1:
            self.local_params = params_sharded[start:]
        else:
            self.local_params = params_sharded[start: start + (p_size // self.world_size)]

        self.optimizer = optimizer_cls(self.local_params, **kwargs)

        # 初始化父类对象
        super().__init__(params_sharded, kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.optimizer.step(closure)

        # 将计算好的参数扩散出去
        # 扩散时由于是多进程，需要进行某种操作使得进程之间没有互锁进而引发严重的死锁
        # 本作业采用较为简单的策略，从头遍历所有参数进行参数梯度传递
        # 对于同一个dist通信代码，所有有关的进程必须全部存在这行代码, 之前的fsdp中的通信进程已经有这个保证了
        for p in range(self.p_size):
            # 这里区分是因为在本策略中，最后一个节点可能承担多个param，这样会导致计算src的时候会超出self.rank的范围（可以用11个p，5个节点举例）
            if p < self.world_size * (self.p_size // self.world_size) :
                dist.broadcast(
                    self.params[p].data,
                    src = p // (self.p_size // self.world_size),
                )
            else:
                dist.broadcast(
                    self.params[p].data,
                    src= self.world_size - 1,
                )

        return loss


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    return optimizer_shared(params, optimizer_cls, **kwargs)
    raise NotImplementedError
