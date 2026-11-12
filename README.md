# ML-KAN-U-Net

本项目实现“统一 1D/2D U-Net 主线 + 多层级 KAN 特征校正 + 离散物理约束 + 跨主干验证”。一维和二维模型仅改变卷积、池化与上采样维度，编码器通道统一为 32、64、128，瓶颈通道为 256。

## 1. 数据真实性与适用范围

本项目中的“真实数据”是由明确 PDE、参数、边界条件和数值求解器产生的数值基准真值，不包含随机生成的目标解：

| 任务 | 真值生成方法 | 主要参数 |
| --- | --- | --- |
| Burgers | 2/3 去混叠 Fourier 伪谱法与 RK4 | ν=1e-2、1e-3、1e-4 |
| Allen–Cahn | 半隐式 Fourier IMEX | ε=0.01、0.02、0.04 |
| Trig-n4 | 与五点差分一致的离散制造解 | 高频主模态 n=4 |
| Peak | 与五点差分一致的局部峰值制造解 | 峰值位置和宽度变化 |
| Exp-Trig | 与五点差分一致的指数—振荡制造解 | 指数系数变化 |
| Gaussian heat | 精确 Fourier 热半群 | α=0.005、0.01、0.02 |

时间依赖方程保存相邻时间层 `(u^n, parameter, coordinates, dt) -> u^(n+1)`，因此反向 Euler 型离散物理残差具有明确含义。训练集、验证集和测试集按照轨迹或参数实例编号划分为 70%/15%/15%，同一轨迹不会跨集合泄漏。

`sample_data` 包含六个由上述求解器实际生成的小规模数据文件，用于验证数据格式和代码。它们不是论文正式结果。论文规模数据由配置文件在服务器上生成。

## 2. 模型编号

| 编号 | 模型 |
| --- | --- |
| M0 | U-Net |
| M1 | U-Net + 单层 KAN-Middle |
| M2 | U-Net + 三级 KAN，固定门控 |
| M3 | U-Net + 三级 KAN，空间门控 |
| M4 | U-Net + 三级 KAN，SE 门控 |
| M5 | ML-KAN-U-Net，空间—通道联合门控 |
| M6 | U-Net + 参数匹配卷积校正 |
| M7 | U-Net + 参数匹配 MLP 校正 |
| M8 | ResCNN |
| M9 | ResCNN + 多阶段 KAN |
| M10 | FNO |
| M11 | FNO + 离散物理约束 |

KAN 默认采用三阶 B 样条，也保留 RBF 基函数选项。三级校正使用独立参数，校正结构为 `1×1降维 -> KAN -> 1×1升维`，内部宽度为 8、16、32，γ 为 0.02、0.05、0.08。

## 3. 安装

```bash
conda env create -f environment.yml
conda activate ml-kan-unet
```

## 4. 数据生成与校验

```bash
python scripts/generate_data.py \
  --task burgers --output data/generated/burgers_N1024.npz \
  --n 1024 --groups 100 --seed 2026 --final-time 0.1 --save-dt 0.01

python scripts/generate_data.py \
  --task trig_n4 --output data/generated/trig_n4_N64.npz \
  --n 64 --groups 1000 --seed 2026

python scripts/validate_dataset.py data/generated/trig_n4_N64.npz
```

每个数据文件同时生成 `manifest.json`，记录方程、求解器、参数、通道、划分、取值范围和 SHA-256 校验值。

## 5. 推荐运行顺序

第一阶段只运行 Burgers 和 Trig-n4 上的 M0、M1、M2，每个模型 5 个种子：

```bash
bash run_stage1.sh
```

阈值结论位于：

```text
results/stage1_summary/decision.json
```

只有结果支持继续扩展时，再运行完整模型和消融矩阵：

```bash
bash run_full.sh
```

所有脚本支持断点续跑，已存在 `metrics.json` 的实验会跳过。

完整实验结束后，也可以单独重新生成全部论文图表：

```bash
bash run_paper_assets.sh
```

该脚本固定生成：

- `results/paper_figures/fig01_*.png/.pdf` 至 `fig14_*.png/.pdf`；
- `results/paper_tables/table01_*.csv/.tex` 至 `table14_*.csv/.tex`。

图1至图3由矢量绘图代码直接构建。图4至图13只读取真实预测、消融、滚动、门控和效率结果。图14读取训练过程中逐轮记录的三级校正量、空间门控、通道门控和有效校正尺度。

## 6. 单模型运行

```bash
python scripts/train_model.py \
  --dataset data/generated/trig_n4_N64.npz \
  --model M2 --config configs/default.json --seed 0 \
  --out results/debug/trig_n4/M2/seed_0
```

## 7. 跨分辨率与鲁棒性

先分别生成 N=64 和 N=128 的同类数据，再执行：

```bash
python scripts/evaluate_generalization.py \
  --run results/main/trig_n4/M2_multilevel_fixed/seed_0 \
  --dataset data/generated/trig_n4_N128.npz \
  --out results/cross_resolution/trig_n4_M2_64_to_128.json
```

输入噪声实验增加 `--noise-std 0.01`。该参数表示相对于首个物理输入通道标准差的噪声比例。

## 8. 门控与校正量导出

```bash
python scripts/export_correction_maps.py \
  --run results/main/trig_n4/M5_joint/seed_0 \
  --dataset data/generated/trig_n4_N64.npz \
  --out results/analysis/trig_n4_gate_maps.npz
```

导出文件包含各层原始特征、校正量、空间门控和校正范数，可用于绘制浅层、中层和深层热力图。

## 9. 统计与结果文件

每次训练输出：

- `metrics.json`：误差、参数量、FLOPs、训练和推理时间；
- `per_sample_metrics.npz`：每个测试样本的误差；
- `history.csv`：全部损失项；
- `best.weights.h5`：验证集最优权重；
- `predictions_preview.npz`：真实解、预测解和输入预览。

汇总脚本计算均值±标准差、bootstrap 95% 置信区间、配对检验、Cohen's dz 和测试样本改善比例。

