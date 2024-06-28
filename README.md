# MUNet++大尺度遥感影像水路识别

## 项目简介

本项目主要关注大尺度遥感影像的河流主体水路的变化，通过遮挡和混合的方法，对训练数据进行在线增强。从而使得模型能习得去抑制一些灌溉、水田、水体养殖等人工水体区域。突出陆地区域以及河流水道的连贯性。使得整体模型能在大尺度遥感影像的河流岸线上有着比WatNet更优的自然岸线识别效果。

*这是一个利用smp神经网络框架，搭建mobilenet_v2为主干的UNet++模型，云遮挡数据增强策略，实现突出自然岸线的水体识别项目。*

## 安装指南

克隆本仓库，依据requirements.txt安装相关依赖：

```bash
git clone https://github.com/Mikky574/MUPP.git
cd MUPP
pip install -r requirements.txt
```

## 使用方式

首先，从本项目的packages中，下载模型权重文件best.pth，放到克隆好的MUPP文件夹下。随后修改interface.py脚本中的路径信息，指定模型权重.pth的路径。指定原始tif数据所在的文件夹，以及输出结果保存的文件夹路径。模型会自动遍历文件夹下的所有.tif后缀的文件，进行水体推理。运行代码如下：

```bash
python interface.py
```

项目适配tif文件和npy文件两种接口，支持格式为Sentinel-2的遥感影像2，3，4，8，11与12六个通道的数据，或是12通道的S2遥感tif数据，自动化推理程序会自动选取第2，3，4，8，11，12通道的数据进行处理。处理包括z-score标准化以及滑动切片，npy格式的数据为tif数据经过标准化处理后的结果。

## 训练模型

用于训练的数据集来源于 [WatNet团队](https://github.com/xinluo2018/WatNet) 提供的 [ESWD地表水数据集](https://zenodo.org/records/5205674)，将下载的数据集文件解压到.\train_data\dset-s2下,随后运行preprocess_files.py脚本，自动处理原始数据为npz格式，以及统计全局均值和方差数据，保存五折训练集路径文件：

```
python .\prepocess\preprocess_files.py -r ".\data\dset-s2" -o ".\data\npz"
```

运行预处理代码主要是需要生成train_pieces文件夹。随后，可以更改train.py中的超参数设置，重新设定train_pieces的路径。指定fold_number五折的折数（可选值1~5）：

```
python train.py
```

## 技术细节

主要是运用云遮挡以及批次混合进行数据集增强，相关的数据处理源码位于train.py当中，优化器、学习率衰减等参考nnUNet中策略进行设置。

## 与WatNet的效果对比

![效果对比图]( figure\图片1.png "对比图")

a: Sentinel-2珠江口遥感影像假彩色波段; 

b: NDWI水体系数归一化灰度图; 

c: WatNet模型推理水陆二值化掩码图; 

d: 我们团队的MUPP推理水陆二值化掩码图。

单在水体识别上，我们的工作并没有超过原有的WatNet模型工作。WatNet模型突出地表水的识别，我们的MUPP突出自然水路信息的识别，抑制了部分水田、灌溉的水体区域。这张遥感图，我们也放在了packages项目包内。
