
```bash
EAI_Lab2/
├─ models/
│  └─ resnet.py            # 可以根據不同的 cfg 建立剪枝前與剪枝後的 ResNet50 模型
├─ sparsity_train.ipynb    # 對 BN 的 scaling factor 進行 sparsity training
├─ modelprune.ipynb        # 根據 \gemma 的大小建立 mask，產生新的 cfg，並真正建立 channel 數較少的模型
└─ train_prune_model.ipynb # 對剪枝後的模型進行 fine-tuning，以恢復準確率
```

## Problem1
Show and compare the test accuracy and model parameters among the original and fine-tuned models.

| Model               | Parameters   | Param Reduction (%)   | FLOPs       | FLOPs Reduction (%)   | Test Accuracy (%)   | Acc vs Original (%)   |
|:--------------------|:-------------|:----------------------|:------------|:----------------------|:--------------------|:----------------------|
| Original ResNet50   | 23,513,162   | -                     | 328,947,712 | -                     | 90.43%              | -                     |
| Pruned (Before FT)  | 6,789,697    | 71.12%                | 191,762,932 | 41.70%                | 90.51%              | +0.08%                |
| Pruned (Fine-tuned) | 6,789,697    | 71.12%                | 191,762,932 | 41.70%                | 91.36%              | +0.93%                |

結論：
- 透過 Network Slimming 剪枝，模型參數量由 23.51M 下降到 6.79M，減少了約 71.12%，FLOPs 從 328.95M 降至 191.76M，下降了約 41.70%，有效地達成了輕量化與推論加速
- 單純剪枝 (尚未 fine-tuning) 就可以發現 test accuracy 有稍微提升 ($+0.08\%$)，驗證了 ResNet50 有 overparameterized 的現象，透過剪枝可以移除雜訊，使泛化能力進一步提升

## Problem2
Please think about why it prompts you to fix the input and output channel numbers of each bottleneck to the channel numbers before pruning. What could happen if they are not fixed, and why?

bottleneck 指的是在神經網路的資料流路徑上，刻意將特徵維度（通道數或隱藏層維度）先大幅壓縮、再重新擴張的一種架構設計模式。在 ResNet50 中，bottleneck 內部包含了一條 shortcut，為了對齊 channel 維度，所以需要去作修正，否則 channel 維度對不上，在前向傳播會出現張量形狀不匹配的錯誤。

## Problem3
Describe the problems you encountered and how you solved them.

BN layer 的 pruning 需要配合 ResNet50 的結構，不能直接 pruning，又因為高維的張量很難想像形狀，常常維度會對不上。解決的方式就是要深入了解 ResNet50；coding 時不要想像，直接推論張量形狀來作對齊。