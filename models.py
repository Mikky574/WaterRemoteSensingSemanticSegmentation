import torch
from torch import nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp

class UNetPlusPlus:
    def __init__(self, model_name):
        self.model_name = model_name
        self.encoder_depth = 5
        self.decoder_channels = [512,256, 128, 64, 32]

        self.model = self._create_model()

    def _create_model(self):
        # 创建 Unet++ 模型
        model = smp.UnetPlusPlus(
            encoder_name=self.model_name,
            encoder_weights="imagenet",
            in_channels=6,
            classes=1,
            encoder_depth=self.encoder_depth,
            decoder_channels=self.decoder_channels,
            activation ="sigmoid",
        )

        return model

    def get_model(self):
        # 返回模型实例
        return self.model


# class P_UNet(nn.Module):  # 加上Res的深监督模型

#     def __init__(self):
#         super().__init__()
#         self.init = nn.Conv2d(6, 16, 1)
#         # (16,256,256)
#         # 修改为ResNet的编码块
#         self.resNet1 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(16, 32, 1),
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#         )
#         self.down_block1 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(16, 16, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(16),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(16, 32, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(32, 32, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         # (32,128,128)
#         self.resNet2 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(32, 64, 1),
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#         )
#         self.down_block2 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(32, 32, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         # (64,64,64)
#         self.resNet3 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(64, 128, 1),
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#         )
#         self.down_block3 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(64, 64, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(64, 128, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(128, 128, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         # (128,32,32)
#         self.resNet4 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(128, 256, 1),
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#         )
#         self.down_block4 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(128, 128, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(128, 256, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(256, 256, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         self.resNet5 = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(256, 512, 1),
#             nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#         )
#         self.down_block5 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(256, 256, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(256, 512, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(512, 512, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         # 第一层的预测，加上注意力机制试试
#         self.predictor0 = nn.Sequential(
#             nn.Conv2d(512, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))

#         self.up_block1 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False),
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         # 增加对应层下采样结果
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(512, 256, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )

#         self.predictor1 = nn.Sequential(
#             nn.Conv2d(256, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.up_block2 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )

#         self.predictor2 = nn.Sequential(
#             nn.Conv2d(128, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.up_block3 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )
#         self.predictor3 = nn.Sequential(
#             nn.Conv2d(64, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.up_block4 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False),
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 32, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )
#         self.predictor4 = nn.Sequential(
#             nn.Conv2d(32, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.up_block5 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(32, 16, 2, stride=2, bias=False),
#             nn.BatchNorm2d(16),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(32, 16, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(16),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )
#         self.predictor5 = nn.Sequential(
#             nn.Conv2d(16, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.conv_list = nn.ModuleList([
#             nn.Conv2d(16 << input_channels, 2, 1)
#             for input_channels in reversed(range(6))
#         ])

#     def attention(self, up_features, prediction_list):  # 注意力机制
#         int_features_list = [
#             F.interpolate(func(feature),
#                           size=(256, 256),
#                           mode='bilinear',
#                           align_corners=True)
#             for func, feature in zip(self.conv_list, up_features)
#         ]
#         int_features = torch.cat(int_features_list, dim=1)
#         attn = F.softmax(int_features, dim=1)
#         predictions = [
#             F.interpolate(prediction,
#                           size=(256, 256),
#                           mode='bilinear',
#                           align_corners=True) for prediction in prediction_list
#         ]
#         predictions = torch.cat(predictions, dim=1)
#         output = torch.sum(attn * predictions, dim=1)
#         output.unsqueeze_(dim=1)
#         return output

#     def forward(self, x):
#         x0 = self.init(x)  # (16,256,256)
#         # res
#         x1 = self.resNet1(x0) + self.down_block1(x0)
#         # (32,128,128)
#         # res
#         x2 = self.resNet2(x1) + self.down_block2(x1)
#         # (64,64,64)
#         # res
#         x3 = self.resNet3(x2) + self.down_block3(x2)
#         # (128,32,32)
#         # res
#         x4 = self.resNet4(x3) + self.down_block4(x3)  # (256,16,16)
#         # res
#         x = self.resNet5(x4) + self.down_block5(x4)
#         # 这里作为上采样的res# (512,8,8)
#         up_features = [x]
#         prediction_list = [self.predictor0(x)]
#         x = self.up_block1(x)
#         x = torch.cat([x, x4], dim=1)
#         x = self.conv1(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor1(x))
#         x = self.up_block2(x)
#         x = torch.cat([x, x3], dim=1)
#         x = self.conv2(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor2(x))
#         x = self.up_block3(x)
#         x = torch.cat([x, x2], dim=1)
#         x = self.conv3(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor3(x))
#         x = self.up_block4(x)
#         x = torch.cat([x, x1], dim=1)
#         x = self.conv4(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor4(x))
#         x = self.up_block5(x)
#         x = torch.cat([x, x0], dim=1)
#         x = self.conv5(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor5(x))
#         output = self.attention(up_features, prediction_list)
#         return output, prediction_list


# class Hed_UNet(nn.Module):  # 深监督的模型

#     def __init__(self):
#         super().__init__()
#         self.init = nn.Conv2d(6, 16, 1)
#         # (16,256,256)
#         self.down_block1 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(16, 16, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(16),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(16, 32, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(32, 32, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         # (32,128,128)
#         self.down_block2 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(32, 32, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         # (64,64,64)
#         self.down_block3 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(64, 64, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(64, 128, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(128, 128, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         # (128,32,32)
#         self.down_block4 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(128, 128, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(128, 256, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(256, 256, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         self.down_block5 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(256, 256, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 增加特征通道
#             nn.Conv2d(256, 512, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # 特征提取
#             nn.Conv2d(512, 512, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(512),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(inplace=False),
#             # nn.Dropout(0.3),
#         )
#         # 第一层的预测，加上注意力机制试试
#         self.predictor0 = nn.Sequential(
#             nn.Conv2d(512, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.up_block1 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False),
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         # 增加对应层下采样结果
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(512, 256, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(256),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )
#         self.predictor1 = nn.Sequential(
#             nn.Conv2d(256, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.up_block2 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(128),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )
#         self.predictor2 = nn.Sequential(
#             nn.Conv2d(128, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.up_block3 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )
#         self.predictor3 = nn.Sequential(
#             nn.Conv2d(64, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.up_block4 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False),
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 32, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )
#         self.predictor4 = nn.Sequential(
#             nn.Conv2d(32, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.up_block5 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(32, 16, 2, stride=2, bias=False),
#             nn.BatchNorm2d(16),
#             # nn.ReLU(inplace=True)
#             nn.ReLU(),
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(32, 16, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(16),
#             # nn.ReLU(inplace=True),
#             nn.ReLU(),
#         )
#         self.predictor5 = nn.Sequential(
#             nn.Conv2d(16, 2, 1),  # 加入预测层效果
#             nn.BatchNorm2d(2))
#         self.conv_list = nn.ModuleList([
#             nn.Conv2d(16 << input_channels, 2, 1)
#             for input_channels in reversed(range(6))
#         ])

#     def attention(self, up_features, prediction_list):  # 注意力机制
#         # def int_conv(x,input_channels):
#         #     return nn.Conv2d(input_channels,1,1)(x)
#         # int_features_list=[]
#         # for i in range(5):
#         int_features_list = [
#             F.interpolate(func(feature),
#                           size=(256, 256),
#                           mode='bilinear',
#                           align_corners=True)
#             for func, feature in zip(self.conv_list, up_features)
#         ]
#         # F.interpolate(int_conv(up_features[i],64<<i), size=(256, 256)
#         # , mode='bilinear', align_corners=True)
#         # int_features_list.append(int_features)
#         int_features = torch.cat(int_features_list, dim=1)
#         attn = F.softmax(int_features, dim=1)
#         predictions = [
#             F.interpolate(prediction,
#                           size=(256, 256),
#                           mode='bilinear',
#                           align_corners=True) for prediction in prediction_list
#         ]
#         predictions = torch.cat(predictions, dim=1)
#         output = torch.sum(attn * predictions, dim=1)
#         output.unsqueeze_(dim=1)
#         return output

#     def forward(self, x):
#         x0 = self.init(x)  # (16,256,256)
#         x1 = self.down_block1(x0)
#         # (32,128,128)
#         x2 = self.down_block2(x1)
#         # (64,64,64)
#         x3 = self.down_block3(x2)
#         # (128,32,32)
#         x4 = self.down_block4(x3)
#         # (256,16,16)
#         x = self.down_block5(x4)
#         # 这里作为上采样的res# (512,8,8)
#         up_features = [x]
#         prediction_list = [self.predictor0(x)]
#         x = self.up_block1(x)
#         x = torch.cat([x, x4], dim=1)
#         x = self.conv1(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor1(x))
#         x = self.up_block2(x)
#         x = torch.cat([x, x3], dim=1)
#         x = self.conv2(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor2(x))
#         x = self.up_block3(x)
#         x = torch.cat([x, x2], dim=1)
#         x = self.conv3(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor3(x))
#         x = self.up_block4(x)
#         x = torch.cat([x, x1], dim=1)
#         x = self.conv4(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor4(x))
#         x = self.up_block5(x)
#         x = torch.cat([x, x0], dim=1)
#         x = self.conv5(x)
#         up_features.append(x)
#         prediction_list.append(self.predictor5(x))
#         output = self.attention(up_features, prediction_list)
#         return output, prediction_list


# class UNet(nn.Module):
    
#     def __init__(self):
#         super().__init__()
#         self.init = nn.Conv2d(6, 64, 1)
#         self.down_block1 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(64, 64, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         # (32,128,128)
#         self.down_block2 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(64, 64, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         # (64,64,64)
#         self.down_block3 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(64, 64, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         # (128,32,32)
#         self.down_block4 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(64, 64, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         self.down_block5 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(64, 64, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         self.up_block1 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(64, 64, 2, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         # 增加长连接
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.up_block2 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(64, 64, 2, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.up_block3 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(64, 64, 2, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.up_block4 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(64, 64, 2, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.up_block5 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(64, 64, 2, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.predictor = nn.Sequential(
#             nn.Conv2d(64, 1, 1),  # 加入预测层效果
#             nn.BatchNorm2d(1))  # 输出一个层，反正是二分类问题，方便点啦

#     def forward(self, x):
#         x0 = self.init(x)  # (16,256,256)
#         x1 = self.down_block1(x0)
#         # (32,128,128)
#         x2 = self.down_block2(x1)
#         # (64,64,64)
#         x3 = self.down_block3(x2)
#         # (128,32,32)
#         x4 = self.down_block4(x3)  # (256,16,16)
#         x = self.down_block5(x4)
#         # (512,8,8)
#         x = self.up_block1(x)
#         x = torch.cat([x, x4], dim=1)
#         x = self.conv1(x)
#         x = self.up_block2(x)
#         x = torch.cat([x, x3], dim=1)
#         x = self.conv2(x)
#         x = self.up_block3(x)
#         x = torch.cat([x, x2], dim=1)
#         x = self.conv3(x)
#         x = self.up_block4(x)
#         x = torch.cat([x, x1], dim=1)
#         x = self.conv4(x)
#         x = self.up_block5(x)
#         x = torch.cat([x, x0], dim=1)
#         x = self.conv5(x)
#         x = self.predictor(x)
#         return x


# class UNet(nn.Module):  # 长连接的UNet模型，不带中心裁剪，方便一点

#     def __init__(self):
#         super().__init__()
#         self.init = nn.Conv2d(6, 16, 1)
#         self.down_block1 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(16, 16, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(16, 32, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(32, 32, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         # (32,128,128)
#         self.down_block2 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(32, 32, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(32, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(64, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         # (64,64,64)
#         self.down_block3 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(64, 64, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(64, 128, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(128, 128, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         # (128,32,32)
#         self.down_block4 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(128, 128, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(128, 256, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(256, 256, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         self.down_block5 = nn.Sequential(
#             # 下采样
#             nn.Conv2d(256, 256, 2, stride=2, bias=False),  # 中间的层统一不添加偏置参数
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             # 增加特征通道
#             nn.Conv2d(256, 512, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             # 特征提取
#             nn.Conv2d(512, 512, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.3),
#         )
#         self.up_block1 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True))
#         # 增加长连接
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(512, 256, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#         )
#         self.up_block2 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, padding=1, padding_mode='zeros',
#                       bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )
#         self.up_block3 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.up_block4 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(64, 32, 2, stride=2, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True))
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(64, 32, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(32), nn.ReLU(inplace=True))
#         self.up_block5 = nn.Sequential(
#             # 上采样
#             nn.ConvTranspose2d(32, 16, 2, stride=2, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True))
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(32, 16, 3, padding=1, padding_mode='zeros', bias=False),
#             nn.BatchNorm2d(16), nn.ReLU(inplace=True))
#         self.predictor = nn.Sequential(
#             nn.Conv2d(16, 1, 1),  # 加入预测层效果
#             # nn.BatchNorm2d(1),
#             nn.Sigmoid()
#             )  # 输出一个层，反正是二分类问题，方便点啦

#     def forward(self, x):
#         x0 = self.init(x)
#         x1 = self.down_block1(x0)
#         x2 = self.down_block2(x1)
#         x3 = self.down_block3(x2)
#         x4 = self.down_block4(x3)
#         x = self.down_block5(x4)
#         x = self.up_block1(x)
#         x = torch.cat([x, x4], dim=1)
#         x = self.conv1(x)
#         x = self.up_block2(x)
#         x = torch.cat([x, x3], dim=1)
#         x = self.conv2(x)
#         x = self.up_block3(x)
#         x = torch.cat([x, x2], dim=1)
#         x = self.conv3(x)
#         x = self.up_block4(x)
#         x = torch.cat([x, x1], dim=1)
#         x = self.conv4(x)
#         x = self.up_block5(x)
#         x = torch.cat([x, x0], dim=1)
#         x = self.conv5(x)
#         x = self.predictor(x)
#         return x


# class UNet(nn.Module):
#     def __init__(self, base_channels=64):
#         super(UNet, self).__init__()
#         self.base_channels = base_channels

#         self.init = nn.Conv2d(6, base_channels, kernel_size=1)

#         # 下采样层
#         self.down_block1 = self._down_block(base_channels, base_channels * 2)
#         self.down_block2 = self._down_block(base_channels * 2, base_channels * 4)
#         self.down_block3 = self._down_block(base_channels * 4, base_channels * 8)
#         self.down_block4 = self._down_block(base_channels * 8, base_channels * 16)

#         # Bottleneck
#         self.bottleneck = self._down_block(base_channels * 16, base_channels * 32)

#         # 上采样层
#         self.up_block = self._up_block(base_channels * 32, base_channels * 16)
#         self.up_block1 = self._up_block(base_channels * 16, base_channels * 8)
#         self.up_block2 = self._up_block(base_channels * 8, base_channels * 4)
#         self.up_block3 = self._up_block(base_channels * 4, base_channels * 2)
#         self.up_block4 = self._up_block(base_channels * 2, base_channels)

#         # 最终卷积层
#         self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
#         self.final_activation = nn.Sigmoid()

#     def _down_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def _up_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(in_channels//2),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(in_channels//2, out_channels, kernel_size=2, stride=2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         # 下采样
#         x = self.init(x)
#         x1 = self.down_block1(x)
#         x2 = self.down_block2(x1)
#         x3 = self.down_block3(x2)
#         x4 = self.down_block4(x3)

#         # Bottleneck
#         x = self.bottleneck(x4)

#         # 上采样和长连接
#         x = self.up_block(x)
        
#         x = torch.cat((x, x4), dim=1)
#         x = self.up_block1(x)
#         x = torch.cat((x, x3), dim=1)
#         x = self.up_block2(x)
#         x = torch.cat((x, x2), dim=1)
#         x = self.up_block3(x)
#         x = torch.cat((x, x1), dim=1)
#         x = self.up_block4(x)

#         x = self.final_conv(x)
#         x = self.final_activation(x)
#         return x


# class UNet:
#     def __init__(self, model_name):
#         self.model_name = model_name
#         self.encoder_depth = 4
#         self.decoder_channels = [256, 128, 64, 32]

#         self.model = self._create_model()

#     def _create_model(self):
#         # 创建 Unet++ 模型
#         model = smp.Unet(
#             encoder_name=self.model_name,
#             encoder_weights="imagenet",
#             in_channels=6,
#             classes=1,
#             encoder_depth=self.encoder_depth,
#             decoder_channels=self.decoder_channels,
#             activation ="sigmoid",
#         )
        
#         # 修改 segmentation_head 以包括 Sigmoid 激活函数
#         # model.segmentation_head = nn.Sequential(
#         #     nn.Conv2d(self.decoder_channels[-1], self.num_classes, kernel_size=3, padding=1),
#         #     nn.Sigmoid()
#         # )

#         return model

#     def get_model(self):
#         # 返回模型实例
#         return self.model
