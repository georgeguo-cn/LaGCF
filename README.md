# LaGCF

This is my Pytorch implementation for the paper:
>Zhiqiang Guo, Chaoyang Wang, Zhi Li, Jianjun Li, Guohui Li(2021). Joint Locality Preservation and Adaptive Combination for Graph Collaborative Filtering, [Paper in Springer](https://link.springer.com/chapter/10.1007/978-3-031-00126-0_12). In DASFAA 2021.

## Configuration.

We run this code on the follwing hardware configuration:
* CPU: Intel Core i7-6850K 3.60GHz, 6 core
* GPU: RTX 3090 24G
* CUDA: cudatoolkit9.0
* cudnn: cudnn7.6.5

The required packages are as follows:
* Python  == 3.7
* Pytorch == 11.3

## Run.

You can train our model by running "python main.py" with default parameters.
```
python main.py
```

Moreover, you also can set different parameters to train it.
