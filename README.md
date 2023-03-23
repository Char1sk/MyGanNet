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

## 下步工作

- 提升模型效果
  - 损失函数(材质/线条/权重等)
  - 网络结构调整(层数/特征融合方式)
  - 调参）
- 评价和比较模型
  - 效果指标(FID FSIM SSIM NLDA 人工二选一)
  - 效率指标(参数量 计算量 访存量 内存占用 推理时间)
  - 其他模型
