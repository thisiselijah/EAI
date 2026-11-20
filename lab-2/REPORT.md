
```bash
EAI_Lab2/
├─ models/
│  └─ resnet.py            # 可以根據不同的 cfg 建立剪枝前與剪枝後的 ResNet50 模型
├─ sparsity_train.ipynb    # 對 BN 的 scaling factor 進行 sparsity training
├─ modelprune.ipynb        # 根據 \gemma 的大小建立 mask，產生新的 cfg，並真正建立 channel 數較少的模型
└─ train_prune_model.ipynb # 對剪枝後的模型進行 fine-tuning，以恢復準確率
```

## Problem1
Show and compare the test accuracy and model parameters among the original and fine-tuned models

## Problem2
Please think about why it prompts you to fix the input and output channel numbers of each bottleneck to the channel numbers before pruning. What could happen if they are not fixed, and why?

## Problem3
Describe the problems you encountered and how you solved them