# You Only Watch Once (YOWO)
## Reimplementation based on [PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo) 


## 1 简介
![YOWO](https://github.com/zwtu/YOWO-Paddle/blob/main/images/YOWO.png?raw=true)

YOWO架构是一个具有两个分支的单阶段网络。一个分支通过2D-CNN提取关键帧（即当前帧）的空间特征，而另一个分支则通过3D-CNN获取由先前帧组成的剪辑的时空特征。为准确汇总这些特征，YOWO使用了一种通道融合和关注机制，最大程度地利用了通道间的依赖性。最后将融合后的特征进行帧级检测。<strong>[[可视化结果]](#使用预测引擎推理)</strong>

<strong>Paper:</strong> Köpüklü O, Wei X, Rigoll G. [You only watch once: A unified cnn architecture for real-time spatiotemporal action localization](https://arxiv.org/pdf/1911.06644.pdf)[J]. arXiv preprint arXiv:1911.06644, 2019.

<strong>Code Reference：</strong>[https://github.com/wei-tim/YOWO](https://github.com/wei-tim/YOWO)

在此非常感谢[Okan Köpüklü](https://github.com/okankop)等人贡献的[YOWO](https://github.com/wei-tim/YOWO)，提高了本repo复现论文的效率。

<strong>复现目标：</strong>UCF101-24数据集，YOWO (16-frame)模型，frame-mAP under IoU threshold of 0.5=80.4 

<strong>AI Studio体验教程</strong>: [基于 PaddleVideo 的YOWO复现](https://aistudio.baidu.com/aistudio/projectdetail/4033737?contributionType=1)

## 2 复现精度
| Model    |3D-CNN backbone | 2D-CNN backbone | Dataset  |Input    | Target <br> Frame-mAP <br>(@ IoU 0.5)    | Our <br> Frame-mAP <br>(@ IoU 0.5)   
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: 
| YOWO | 3D-ResNext-101 | Darknet-19 | UCF101-24 | 16-frames, d=1 | <strong>80.40</strong>| <strong>80.83</strong> 

- 模型权重：[百度网盘](https://pan.baidu.com/s/1RznZpYpZxoZg9XiUWNkQWQ?pwd=r5g3) | [飞桨 AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/146682)

## 3 数据集和预训练权重

### 3.1 下载数据集 
[Google Drive](https://drive.google.com/file/d/1o2l6nYhd-0DDXGP-IPReBP4y1ffVmGSE/view) | [飞桨 AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/36600)，存放路径 ``` data/ucf24/ ```

### 3.2 生成所需格式的 annotations

```
python data/ucf24/build_split.py
```

### 3.3 下载预训练权重

[飞桨 AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/145592)，存放路径 ``` data/ucf24/ ```

## 4 环境依赖

- GPU：Tesla V100 32G
- Framework：PaddlePaddle == 2.2.2
-  ``` pip install -r requirements.txt ```


## 5 快速开始

### 5.1 Clone 本项目

```
git clone https://github.com/zwtu/YOWO-Paddle.git
cd YOWO-Paddle
```

### 5.2 模型训练

- 参数配置文件在 ``` configs/localization ```
  训练完成后，模型参数保存至 ``` output/```

    ```
    python3 main.py -c configs/localization/yowo.yaml --seed=1
    ```

- 部分训练日志：

    ```
    [05/15 19:13:43] epoch:[  5/5  ] train step:42100 loss_avg: 0.64974  lr: 0.000006 nCorrect_avg: 9.3 batch_cost: 0.84909 sec, reader_cost: 0.00030 sec, ips: 9.42181 instance/sec.
    [05/15 19:13:59] epoch:[  5/5  ] train step:42120 loss_avg: 0.64974  lr: 0.000006 nCorrect_avg: 9.3 batch_cost: 0.81470 sec, reader_cost: 0.00029 sec, ips: 9.81956 instance/sec.
    [05/15 19:14:16] epoch:[  5/5  ] train step:42140 loss_avg: 0.64972  lr: 0.000006 nCorrect_avg: 9.3 batch_cost: 0.79969 sec, reader_cost: 0.00031 sec, ips: 10.00387 instance/sec.
    [05/15 19:14:32] epoch:[  5/5  ] train step:42160 loss_avg: 0.64966  lr: 0.000006 nCorrect_avg: 9.3 batch_cost: 0.81808 sec, reader_cost: 0.00029 sec, ips: 9.77899 instance/sec.
    [05/15 19:14:48] epoch:[  5/5  ] train step:42180 loss_avg: 0.64959  lr: 0.000006 nCorrect_avg: 9.3 batch_cost: 0.82894 sec, reader_cost: 0.00034 sec, ips: 9.65085 instance/sec.
    [05/15 19:15:05] epoch:[  5/5  ] train step:42200 loss_avg: 0.64960  lr: 0.000006 nCorrect_avg: 9.3 batch_cost: 0.81701 sec, reader_cost: 0.00033 sec, ips: 9.79176 instance/sec.
    [05/15 19:15:21] epoch:[  5/5  ] train step:42220 loss_avg: 0.64974  lr: 0.000006 nCorrect_avg: 9.3 batch_cost: 0.82311 sec, reader_cost: 0.00027 sec, ips: 9.71925 instance/sec.
    [05/15 19:15:28] END epoch:5   train loss_avg: 0.64974  nCorrect_avg: 9.3 avg_batch_cost: 0.80803 sec, avg_reader_cost: 0.00026 sec, batch_cost_sum: 34729.53116 sec, avg_ips: 9.72751 instance/sec.
    [05/15 19:15:32] training YOWO finished
    ```


- 如果想在训练期间验证模型的性能，可在运行命令中添加```--validate```，验证时评价指标为```fscore```，与测试模式不同，仅作为参考

    ```
    python3 main.py -c configs/localization/yowo.yaml --validate --seed=1
    ```

- 部分验证日志：

    ```
    [05/15 23:47:15] epoch:[  5/5  ] val step:17100 loss_avg: 0.00000 recall_avg: 0.86527 precision_avg: 0.91334 fscore_avg: 0.87903 batch_cost: 0.73636 sec, reader_cost: 0.00000 sec, ips: 10.86421 instance/sec.
    [05/15 23:47:30] epoch:[  5/5  ] val step:17120 loss_avg: 0.00000 recall_avg: 0.86527 precision_avg: 0.91328 fscore_avg: 0.87902 batch_cost: 0.78743 sec, reader_cost: 0.00000 sec, ips: 10.15966 instance/sec.
    [05/15 23:47:44] epoch:[  5/5  ] val step:17140 loss_avg: 0.00000 recall_avg: 0.86543 precision_avg: 0.91339 fscore_avg: 0.87916 batch_cost: 0.67931 sec, reader_cost: 0.00000 sec, ips: 11.77662 instance/sec.
    [05/15 23:47:58] epoch:[  5/5  ] val step:17160 loss_avg: 0.00000 recall_avg: 0.86555 precision_avg: 0.91344 fscore_avg: 0.87926 batch_cost: 0.70100 sec, reader_cost: 0.00000 sec, ips: 11.41232 instance/sec.
    [05/15 23:48:12] epoch:[  5/5  ] val step:17180 loss_avg: 0.00000 recall_avg: 0.86497 precision_avg: 0.91282 fscore_avg: 0.87867 batch_cost: 0.64150 sec, reader_cost: 0.00000 sec, ips: 12.47071 instance/sec.
    [05/15 23:48:22] END epoch:5   val recall_avg: 0.86427 precision_avg: 0.91207 fscore_avg: 0.87795 avg_batch_cost: 0.51627 sec, avg_reader_cost: 0.00000 sec, batch_cost_sum: 14062.70232 sec, avg_ips: 9.78190 instance/sec.
    ```


### 5.3 模型评估

- 训练日志中记录的验证指标```fscore```不代表最终的测试分数，因此在训练完成之后可以用测试模式对最好的模型进行测试获取最终的指标。评估时评价指标为```Frame-mAP (@ IoU 0.5)```

    ```
    python3 main.py -c configs/localization/yowo.yaml --test --seed=1 -w 'output/YOWO/YOWO_epoch_00005.pdparams'
    ```

- 部分评估日志：
    ```
    [05/18 02:48:33] [TEST] Processing batch 16000/17194 ...
    [05/18 02:50:28] [TEST] Processing batch 16100/17194 ...
    [05/18 02:52:14] [TEST] Processing batch 16200/17194 ...
    [05/18 02:54:07] [TEST] Processing batch 16300/17194 ...
    [05/18 02:55:39] [TEST] Processing batch 16400/17194 ...
    [05/18 02:57:11] [TEST] Processing batch 16500/17194 ...
    [05/18 02:58:46] [TEST] Processing batch 16600/17194 ...
    [05/18 03:00:20] [TEST] Processing batch 16700/17194 ...
    [05/18 03:01:58] [TEST] Processing batch 16800/17194 ...
    [05/18 03:03:31] [TEST] Processing batch 16900/17194 ...
    [05/18 03:05:06] [TEST] Processing batch 17000/17194 ...
    [05/18 03:06:37] [TEST] Processing batch 17100/17194 ...
    /home/aistudio/PaddleVideo-develop/data/ucf24/groundtruths_ucf
    /home/aistudio/PaddleVideo-develop/output/detections_test
    [05/18 03:11:28] AP: 69.70% (1)
    [05/18 03:11:28] AP: 95.83% (10)
    [05/18 03:11:28] AP: 74.26% (11)
    [05/18 03:11:28] AP: 57.32% (12)
    [05/18 03:11:28] AP: 71.73% (13)
    [05/18 03:11:28] AP: 91.40% (14)
    [05/18 03:11:28] AP: 77.28% (15)
    [05/18 03:11:28] AP: 82.29% (16)
    [05/18 03:11:28] AP: 69.48% (17)
    [05/18 03:11:28] AP: 88.85% (18)
    [05/18 03:11:28] AP: 92.02% (19)
    [05/18 03:11:28] AP: 66.70% (2)
    [05/18 03:11:28] AP: 87.56% (20)
    [05/18 03:11:28] AP: 79.10% (21)
    [05/18 03:11:28] AP: 75.66% (22)
    [05/18 03:11:28] AP: 72.89% (23)
    [05/18 03:11:28] AP: 86.22% (24)
    [05/18 03:11:28] AP: 83.10% (3)
    [05/18 03:11:28] AP: 77.31% (4)
    [05/18 03:11:28] AP: 71.02% (5)
    [05/18 03:11:28] AP: 95.35% (6)
    [05/18 03:11:28] AP: 92.35% (7)
    [05/18 03:11:28] AP: 90.77% (8)
    [05/18 03:11:28] AP: 91.80% (9)
    [05/18 03:11:28] mAP: 80.83%
    ```


### 5.4 模型推理

#### 导出inference模型

```
python3 tools/export_model.py -c configs/localization/yowo.yaml -p 'output/YOWO/YOWO_epoch_00005.pdparams'     
```

上述命令将在默认路径```inference/```下生成预测所需的模型结构文件```YOWO.pdmodel```和模型权重文件```YOWO.pdiparams```


#### 使用预测引擎推理

```
python3 tools/predict.py -c configs/localization/yowo.yaml -i 'data/ucf24/HorseRiding.avi' --model_file ./inference/YOWO.pdmodel --params_file ./inference/YOWO.pdiparams
```

输出示例如下（可视化后）:
<div align="center" style="width:image width px;">
  <img  src="https://github.com/zwtu/YOWO-Paddle/blob/main/images/HorseRiding.gif?raw=true" width=240 alt="Horse Riding">
</div>

可以看到，使用在UCF-24上训练好的YOWO模型对```data/ucf24/HorseRiding.avi```进行预测，每张帧输出的类别均为HorseRiding，置信度为0.80左右。

## 6.TIPC

- 安装日志工具
    ```
    pip install https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```

- 运行 ```prepare.sh```

    ```
    bash test_tipc/prepare.sh test_tipc/configs/YOWO/train_infer_python.txt 'lite_train_lite_infer'
    ```

- 运行 ```test_train_inference_python.sh```， 具体参数设置修改于```test_tipc/configs/YOWO/train_infer_python.txt```

    ```
    bash test_tipc/test_train_inference_python.sh test_tipc/configs/YOWO/train_infer_python.txt 'lite_train_lite_infer'
    ```


## 7. 代码结构

```
|-- YOWO-Paddle
    |-- configs                              
        |-- localization
            |-- yowo.yaml                   # 模型训练所需的yaml       
    |-- data                                # 数据集
        |-- ucf24                  
            |-- splitfiles                  # 原始 annotations
            |-- build_split.py              # 生成所需 train/val list
        |-- UCF-24-lite                     # 用于TIPC的小数据集
    |-- output                              # 输出
    |-- paddlevideo                         
        |-- loader                          # 数据集和数据增强
        |-- metrics                         # 评价指标
        |-- modeling                        # 模型定义
        |-- solver                          # 优化器
        |-- task                            # train/test 入口
        |-- utils                           # 工具
    |-- test_tipc                           # TIPC训推一体化认证                         
    |-- tools                               
        |-- export_model.py                 # 导出静态模型
        |-- predict.py                      # 预测 
    |-- main.py                             # 主程序
    |-- README.md                       
    |-- requirements.txt                    # 环境配置文件

```

## 8. LICENSE
本项目的发布受[Apache 2.0 license](https://github.com/zwtu/YOWO-Paddle/blob/main/LICENSE)许可认证。

## 9. 致谢
非常感谢 百度 PaddlePaddle AI Studio 提供的算力支持！

## 我们
团队名称: Team X
学校: NJUST
作者: [zwtu](https://github.com/zwtu) / [keke chen](https://github.com/ping-High) / [zzy](https://github.com/klinic)






