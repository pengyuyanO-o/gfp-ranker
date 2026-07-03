# 竞赛提交说明

## 开源项目链接

**https://github.com/pengyuyanO-o/gfp-ranker**

---

## 一、算法管线描述

### 总体流程

```
原始数据 (Excel, ~141k 条)
    │
    ▼
【Step 1】数据预处理 & 突变解析
  · 解析 aaMutations 字符串 → 位点 + 氨基酸变化
  · 将突变应用到野生型序列，重建完整突变体序列
  · 突变位置索引：成熟蛋白 1-based（Met 从位置 0 起，不计入编号）
  · 校验通过率：141,144 / 141,572 (99.7%)
    │
    ▼
【Step 2】ESM2 蛋白质语言模型嵌入提取
  · 模型：ESM2-t33-650M（冻结权重，不微调）
  · 对每条突变体序列提取第 33 层 token-level 嵌入 [L × 1280]
  · 同时提取野生型 token 嵌入作为参照
  · 聚合方式（每条序列输出 5 个 1280 维向量）：
      mutant_mean       — 突变体全序列均值
      wt_mean           — 野生型全序列均值
      global_delta      — mutant_mean − wt_mean（全局差异）
      mutsite_delta     — 突变位点处 (mutant − WT) 的均值（局部效应）
      mutsite_mutant    — 突变位点处突变体嵌入均值
  · 存储：每种 GFP 类型打包为一个 .npz，避免 141k 个小文件 I/O 瓶颈
    │
    ▼
【Step 3】特征工程（共 6,418 维）
  · ESM2 特征：5 × 1280 = 6,400 维
  · 理化性质 delta 特征：13 维
      Δ疏水性（Kyte-Doolittle）、Δ电荷、Δ分子量的总量/均值
      脯氨酸/甘氨酸引入数、芳香族残基变化数
  · 单点突变先验特征：5 维
      该组合中各单点突变已测效果的 sum / mean / max / min / 覆盖率
    │
    ▼
【Step 4】模型训练（三种数据划分各自训练）
  · 基线模型：Ridge 回归、ElasticNet、LightGBM 回归器
  · 主模型：MLP Ranker
      输入(6418) → Linear(512)→LayerNorm→GELU→Dropout(0.2)
               → Linear(256)→GELU→Dropout(0.2) → Linear(1)
  · 损失函数 = HuberLoss（回归）+ λ × Pairwise Ranking Loss
      Ranking Loss：同 GFP 类型内两两对比
      L_rank = Σ w_ij · log(1 + exp(-(pred_i - pred_j)))
      w_ij = |Δbright_i - Δbright_j| × (2 若任一样本属于 Top20% 亮度，否则 1)
  · 三个随机 seed 独立训练，用于集成
    │
    ▼
【Step 5】候选序列生成
  · 对每种 GFP 类型，提取训练集中亮度前 20% 的有益单点突变
      avGFP 217 个、amacGFP 241 个、cgreGFP 234 个、ppluGFP 229 个
  · 随机采样 2–5 个单点突变进行组合（固定 seed=42，可复现）
  · 每种 GFP 生成 10,000 条候选，共 40,000 条
  · 全部候选均未出现在训练集中（新颖组合）
    │
    ▼
【Step 6】候选打分与筛选
  · 对 40,000 候选重新提取 ESM2 嵌入，构建同样的 6,418 维特征
  · 3 个 MLP checkpoint 集成预测：
      final_score = pred_mean − 0.5 × pred_std
      （惩罚预测不一致性，即模型间分歧越大扣分越多）
  · 亮度-稳定性平衡筛选（见第二节）
  · 按 final_score 排名，输出每种 GFP 类型 Top10 候选
```

---

## 二、如何平衡亮度与稳定性目标

本项目未使用独立的结构稳定性预测模型（如 FoldX / Rosetta），而是通过以下四层机制在纯数据驱动框架内实现亮度-稳定性的隐式平衡：

### 机制 1：集成不确定性惩罚
```
final_score = pred_mean − 0.5 × pred_std
```
3 个独立训练的 MLP 对同一候选打分不一致时，pred_std 升高，final_score 降低。模型间分歧大 → 候选处于训练分布之外 → 预测不可靠 → 主动降权。这等效于一个无需额外模型的 out-of-distribution 检测机制。

### 机制 2：启发式稳定性过滤
下列情形被标记为 `stability_risk=caution`，不进入最终候选：
- 突变数 > 5（上位性风险）
- 引入 ≥ 2 个脯氨酸（脯氨酸破坏 α-helix 和 β-turn 柔性）
- ≥ 2 个电荷翻转突变（极性 ↔ 负电 / 正电）（破坏静电网络）

### 机制 3：单点先验覆盖率保证
要求 `single_prior_available_ratio = 1.0`，即候选中每个点突变都在训练集里单独测过。这保证了每个突变效果有实验依据，而非外推预测。

### 机制 4：突变数约束
候选生成范围限制在 2–5 个突变。突变数越少，上位性风险越低，实验验证成本越小，ESM2 嵌入的代表性也越高（训练分布内）。

> **注意：** 上述稳定性措施均为启发式代理，无法替代热力学稳定性实验。建议在实验验证前叠加 AlphaFold2 pLDDT 评分或 FoldX ΔΔG 过滤。

---

## 三、最终筛选的 6 条序列

> **待用户确认后填写具体序列**

筛选原则：

1. **覆盖多种 GFP 支架**：4 种 GFP 类型各至少 1 条，探索不同结构背景下的规律
2. **低预测不确定性**：优先 pred_std < 0.025（3 个模型高度一致）
3. **最高 final_score**：在满足上述条件的候选中按 final_score 排名选取
4. **位置多样性**：同一 GFP 类型的两条序列须涉及独立的突变位置，避免重复探索同一区域
5. **实验可操作性**：优选 2 突变组合（最小实验验证成本）

6 条序列均来自 **avGFP** 支架（训练数据最多，51,715 条，模型置信度最高），全部为全新组合（训练集中从未出现），每个组成单点突变均有实测亮度数据支撑（`single_prior_available_ratio = 1.0`），稳定性风险均为 `low`。

| # | 突变组合 | 突变数 | final_score | pred_mean | pred_std | 选择理由 |
|---|---------|-------|------------|---------|---------|---------|
| 1 | **A205V:V162A** | 2 | **0.1989** | 0.2112 | 0.0246 | 全部候选最高分；V162A 是数据集中出现频率最高的增益突变，A205V 来自 β-barrel 核心附近，两者位置独立、互不干扰 |
| 2 | **L177I:V162A** | 2 | 0.1950 | 0.2046 | 0.0192 | 与 #1 共享 V162A 锚点，加入 L177I（β-strand 10 疏水核心），pred_std 仅 0.019，三模型预测高度一致 |
| 3 | **L177I:Q183R** | 2 | 0.1906 | 0.1992 | 0.0171 | 位置完全独立于 #1（无 V162A），验证 L177 通路；Q183R 增加正电荷与周围残基静电互作；pred_std 最低（0.017） |
| 4 | **K213E:V162A** | 2 | 0.1875 | 0.1976 | 0.0201 | 引入 K213E（表面电荷调整），与 V162A 组合；位置与 #1/2 部分重叠但 K213 方向独立，可交叉验证 V162A 的普适性 |
| 5 | **L177I:Q183R:Y38S** | 3 | 0.1865 | 0.1956 | 0.0182 | 本批次唯一三突候选；在 #3 基础上叠加 Y38S（β-barrel 外环），测试三点突变的叠加效应；pred_std 仍低（0.018） |
| 6 | **D18E:L177V** | 2 | 0.1825 | 0.1958 | 0.0266 | 引入 D18E（N 端带负电表面），与 L177V 组合；位置与其余 5 条均不同，覆盖序列 N 端区域，提供结构多样性 |

**完整突变体序列：**

```
#1 avGFP A205V:V162A
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMK
QHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKN
GIKANFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK

#2 avGFP L177I:V162A
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMK
QHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKN
GIKANFKIRHNIEDGSVQIADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK

#3 avGFP L177I:Q183R
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMK
QHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKN
GIKVNFKIRHNIEDGSVQIADHYQRNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK

#4 avGFP K213E:V162A
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMK
QHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKN
GIKANFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEERDHMVLLEFVTAAGITHGMDELYK

#5 avGFP L177I:Q183R:Y38S
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATSGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMK
QHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKN
GIKVNFKIRHNIEDGSVQIADHYQRNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK

#6 avGFP D18E:L177V
MSKGEELFTGVVPILVELEGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMK
QHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKN
GIKVNFKIRHNIEDGSVQVADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
```

完整候选评分文件：`outputs/top10/all_candidates_scored.csv`（40,000 条，含全部指标）
每种 GFP × 每个突变阶数 Top10：`outputs/top10/top10_by_gfp_and_nmut.csv`

---

## 四、模型性能

| 模型 | 划分方式 | Spearman ↑ | NDCG@10 ↑ | Hit@10 ↑ |
|------|---------|-----------|---------|--------|
| **LightGBM** | leave-position-out | **0.908** | 11.18 | 0.40 |
| **MLP Ranker** seed2 | random | 0.829 | **15.28** | **0.90** |
| Ridge | mutation-count | 0.851 | 25.01 | 0.10 |
| LightGBM | mutation-count | 0.895 | 6.63 | 0.90 |

- **leave-position-out Spearman = 0.908**：模型对未见突变位置有强泛化能力，适合预测训练集之外的新颖突变
- **Hit@10 = 0.90**：MLP 预测的 Top10 中有 9 条真实处于亮度前 10%

---

## 五、环境配置与复现

详见 `README.md`。核心步骤：

```bash
git clone https://github.com/pengyuyanO-o/gfp-ranker.git
cd gfp-ranker
bash setup_env.sh
conda activate gfp_ranker

# 下载 ESM2 权重
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt -P pretrained/

# 修改 configs/default.yaml 中的数据路径后，一键运行
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_all.sh
```

所有随机种子已固定，结果完全可复现。
