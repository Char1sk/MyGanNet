<!-- 特征融合D？

D, D1, D2, D3
adv_all, adv_parts -->
# 中期

## 复现和改进

- 效果将就，指标比不上
- 做了一些小调整的实验(数据处理/权重/激活)，没有明显变化
- FID: 86.56/95.91 + 41.3 （FID有点不够准
- 309 MB + 10 MB + 2.6184s

## 模型尝试

- train还行，test有点拉，指标差了点（但是练得短）
- 做了基本的训练，还没调参数和结构等
- FID: 41.63/112.69
- 231 MB + 42 MB + 1.0617s (其他的还没比)

## 一些结论

- 总体上观感可，但对背景、头发、脖子拟合不行，但符合场景需求
- MSE比BCE好
- random-size能排除padding，指标更高
- inter比final对Adv好
  - 图像质量更好，割裂感弱
  - 指标没有明显变化(可能是save)
- 对train拟合得更好，部分test-sample依然很差
- ld+下半似乎效果好点，但tb貌似有彩色噪点？
- gc+似乎没有明显优势，可能会过拟合
- DE指标稍稍下降，左右色深一致
- TE指标更下降，全局更接近，更不稳定，体感貌似可
  - all_D并不见得更好，指标还更低，猜想是优化方向混合了

## 下步工作

- 提升模型效果
  - 损失函数(材质/线条/权重等)
  - 网络结构调整(层数/特征融合方式/LLL)
  - 调参）
- 评价和比较模型
  - 效果指标(FID FSIM SSIM NLDA 人工二选一)
  - 效率指标(参数量 计算量 访存量 内存占用 推理时间)
  - 其他模型

## 实验结果

### 对比实验：Base SE DE TE

- 处理背景、随机大小
- 部分D、inter
- L1、MSE、Per

|  Model   | FID  | FSIM | Params | FLOPs | Time |
| :------: | :--: | :--: | :----: | :---: | :--: |
| Baseline |      |      |        |       |      |
|    SE    |      |      |        |       |      |
|    DE    |      |      |        |       |      |
|    TE    |      |      |        |       |      |

- D算不算Params/FLOPs？Time多少？

### 消融实验：data, archi, loss

#### 数据预处理

- Background
- Shape

|      | Background | Shape | FID  |
| :--: | :--------: | :---: | :--: |
| Base |     X      |   X   |      |
| Base |     O      |   X   |      |
| Base |     O      |   O   |      |
|  SE  |     X      |   X   |      |
|  SE  |     O      |   X   |      |
|  SE  |     O      |   O   |      |

#### 损失函数

- L1/L2
- Perceptual
- MSE/BCE

| Adv  | Per  | Com  |  FID  |
| :--: | :--: | :--: | :---: |
| BCE  |  O   |  L1  |   -   |
| MSE  |  X   |  L1  |   -   |
| MSE  |  O   |  L2  |   -   |
| MSE  |  O   |  L1  | 95.97 |

#### 模型架构

- SE DE TE
- all_D/inter

| Model | Dtype  |   FID   |  FSIM  | Params  |   FLOPs   |       Time        |
| :---: | :----: | :-----: | :----: | :-----: | :-------: | :---------------: |
|  SE   | global | 112.7 ? |   -    |    =    |     =     |         =         |
|  SE   | inter  |  101.9  | 0.7771 | 60.52 M | 2*31.28 G | 1.19 s / 0.0444 s |
|  DE   | local  |    -    |   -    |    =    |     =     |         =         |
|  DE   | inter  |  97.52  | 0.7876 | 57.10 M | 2*31.28 G | 1.27 s / 0.0448 s |
|  TE   | local  |  108.5  | 0.7775 |    =    |     =     |         =         |
|  TE   | inter  |  95.97  | 0.7850 | 53.69 M | 2*31.35 G | 1.16 s / 0.0443 s |

