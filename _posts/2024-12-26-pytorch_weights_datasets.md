---
layout: post
title: pytorch_weights_datasets
date: 2024-12-26
categories: [reading]
tags: reading
---
<!--more-->


可以通过下面的方式简单地实现加权数据集的采样

```
class WeightDatasets(Dataset):
    def __init__(self, datasets, p_datasets=None):
        self.datasets = datasets
        if p_datasets is None:
            self.p_datasets = [len(d) for d in self.datasets]
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, index):
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        return dataset.__getitem__(index)
```

然后把这个dataset 传入dataloder即可, 参考ATOM的代码.

