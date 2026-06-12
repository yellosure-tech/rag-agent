# MiniMind 项目笔记

MiniMind 是一个轻量级大语言模型学习项目，覆盖 tokenizer、数据处理、Decoder-only Transformer、预训练、SFT、DPO、LoRA 和推理部署等环节。项目适合用来理解大语言模型从数据到模型再到推理接口的完整工程链路。

## 模型结构

MiniMind 的主体结构是 Decoder-only Transformer。核心模块包括 token embedding、RoPE 旋转位置编码、GQA 注意力机制、SwiGLU 前馈网络、RMSNorm 归一化、权重共享和 lm_head 输出层。推理阶段可以使用 KV-Cache 缓存历史 key/value，减少自回归生成时的重复计算。

## 训练链路

预训练阶段采用 next-token prediction 目标，输入为 `sample[:-1]`，标签为 `sample[1:]`。SFT 阶段使用多轮对话数据，通过 chat template 构造 prompt，并用 loss mask 只对 assistant 回复部分计算损失。DPO 阶段使用 chosen/rejected 偏好样本，让 policy model 更偏向人类偏好的回答。

## 实验记录

tiny 规模实验用于验证训练链路是否打通，而不是追求真实模型能力。实验中保存了 pretrain checkpoint 和 SFT checkpoint，并记录 loss 曲线。pretrain loss 从 8.75 降至 1.67，SFT loss 从 4.40 降至 0.058，说明小数据闭环可以收敛。

