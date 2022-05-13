# You Only Watch Once (YOWO)
## Reimplementation based on [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo) 


## 1. 简介
![YOWO](https://github.com/zwtu/YOWO-Paddle/blob/main/images/YOWO.png?raw=true)

YOWO架构是一个具有两个分支的单阶段网络。一个分支通过2D-CNN提取关键帧（即当前帧）的空间特征，而另一个分支则通过3D-CNN获取由先前帧组成的剪辑的时空特征。为准确汇总这些特征，YOWO使用了一种通道融合和关注机制，最大程度地利用了通道间的依赖性。最后将融合后的特征进行帧级检测。

框架中的3D-CNN架构为3D-ResNext-101，在2D-CNN分支中采用Darknet-19作为基本架构，以并行提取关键帧的二维特征，解决空间定位问题。CFAM模块基于Gram（格拉姆）矩阵来映射通道间的依赖关系，达到融合三维和二维网络输出特征的目的。

<strong>Paper:</strong> Köpüklü O, Wei X, Rigoll G. [You only watch once: A unified cnn architecture for real-time spatiotemporal action localization](https://arxiv.org/pdf/1911.06644.pdf)[J]. arXiv preprint arXiv:1911.06644, 2019.

<strong>Code Reference：</strong>[https://github.com/wei-tim/YOWO](https://github.com/wei-tim/YOWO)

<strong>复现目标：</strong>UCF101-24数据集，YOWO (16-frame)模型，frame-mAP under IoU threshold of 0.5=80.4 


## 2. 复现精度
| Model    |3D-CNN backbone | 2D-CNN backbone | Dataset  |Input    | Target <br> Frame-mAP (@ IoU 0.5)    | Our <br> Frame-mAP (@ IoU 0.5)   
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: 
| YOWO | 3D-ResNext-101 | Darknet-19 | UCF101-24 | 16-frames, d=1 | <strong>80.4</strong>| <strong>80.4</strong> 

- 训练日志：[Log]()
- 模型权重：[Google Drive]()

## 3. 数据集和预训练权重

1. 下载数据集 [Google Drive](https://drive.google.com/file/d/1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE/view) or [飞桨 AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/36600)，存放路径 ``` data/ucf24/ ```

2. 生成所需格式的 annotations

    ```
    cd data/ucf24/
    python build_split.py
    ```

3. 下载预训练权重

    ```
    
    ```


## 4. 环境依赖

- GPU：Tesla V100 32G
- Framework：PaddlePaddle == 2.2.2
-  ``` pip install -r requirements.txt ```


## 5. 快速开始

1. Clone 本项目

    ```
    git clone https://github.com/zwtu/YOWO-Paddle.git
    cd YOWO-Paddle
    ```

2. 模型训练

- 参数配置文件在 ``` configs/localization ```

    ```
    python3 main.py -c configs/localization/yowo.yaml --seed=1
    ```

- [完整训练日志]()
    部分训练日志：
    ```

    ```

    训练完成后，模型参数保存至 ``` output/```