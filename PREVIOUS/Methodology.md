## Methodology

### OpenS2S 中的白盒攻击切入点

- **原始波形 (x)** → **Audio Encoder** → **Speech Adapter** → 与 **text embedding** 拼接 → 输入 **Instruction-Following LLM**

我们的白盒攻击选择的模型位置是再

## 2. 核心算法（替换你原来的“广播注入”公式）

### 2.1 表征定义

设原始语音经过 Audio Encoder（可选再过 Adapter）得到 frame-level hidden states：

$$
H(x)\in \mathbb{R}^{T\times D}
$$

我们用一个稳定的 pooling 得到 utterance-level embedding：

$$
h(x)=\frac{1}{T}\sum_{t=1}^{T}H(x)_t\in\mathbb{R}^{D}
$$

（工程上你现在 D≈1280 是一致的。）

### 2.2 情绪方向 (v) 的构建（找了两个方法来做构建，后续可以看看有没有更好的方法，看起来更科学的方法）

**我们最开始的是用均值差直接构造（不清楚会不会局限于样本）**

$$
\mu_{sad}=\mathbb{E}[h(x)\mid sad],\quad \mu_{happy}=\mathbb{E}[h(x)\mid happy]
$$

$$
v_{mean}=\frac{\mu_{sad}-\mu_{happy}}{|\mu_{sad}-\mu_{happy}|}
$$

**我查到的文献有做probe的work**（这个还没尝试）
训练一个线性情绪分类器，大概可以理解成自己用样本训练一个SER（可以有情绪分类和方向构建）

$$
p(sad\mid h)=\sigma(w^\top h+b)
$$

则情绪方向取：

$$
v_{probe}=\frac{w}{|w|}
$$

### 2.3 构造对抗样本（具体公式目前只是表征，后续优化）

定义投影分数：

$$
s(x)=\langle h(x),v\rangle
$$

我们希望对抗样本 (x') 相比原始 (x) 在 (v) 上显著提升：

$$
\Delta s = s(x')-s(x)
$$

构造Loss**情绪损失**（我们主要通过这个loss改变情绪）

$$
L_{emo}(x')=-(s(x')-s(x))
$$

**保证语义层不做损失，我们构造一个语义的loss**（鉴于之前直接加和对语义层的破坏，加入更稳定的语义）

$$
\Delta h = h(x')-h(x)
$$

$$
\Delta h_{\perp}=\Delta h - \langle \Delta h, v\rangle v
$$

$$
L_{sem}(x')=|\Delta h_{\perp}|_2^2
$$

**扰动控制**到人耳不能察觉（用于物理世界的合理攻击）

$$
L_{per}(x')=|x'-x|_2^2
$$

**总损失**

$$
L(x')=\lambda_{emo}L_{emo}+\lambda_{sem}L_{sem}+\lambda_{per}L_{per}
$$

优化目标上述总loss最小，约束可以加：语义层扰动不大于0，扰动控制小于$\mu$

### 2.4 做迭代优化对抗样本

约束扰动幅度 (

$$
|x'-x|*\infty \le \epsilon
$$

)：

$$
x'*{k+1}=\Pi_{[x-\epsilon,x+\epsilon]}\Big(x'*k-\alpha\cdot \text{sign}(\nabla*{x'}L)\Big)
$$

做一个PGD循环优化直到优化到我们的目标

Following the methodology, here are some ideas about emotional plate：

1. 先用类似散点图的方式，展现图中样本点的集合聚类特征。（难点：高维压缩到2维，同时保持位置特征）
   UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction
2. 计算情绪散点的几何中心点，链接后可得攻击方向向量
3. 投影分数可以用颜色深浅来表示，视觉上直观展现从 Happy 簇移动到 Sad 簇的过程，本质上就是 **$s(x)$** 值不断增加的过程。
4. 最终的情绪色轮
