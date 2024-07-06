import math
import pylab as p
import numpy as np
import timm
import torch
import torch.nn as nn
# from timm.layers import LayerNorm
from torch.nn import init, functional
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from flatten_swin import FLattenSwinTransformer
from einops import rearrange


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num=701, drop_rate=0.4, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [
            nn.Linear(input_dim, num_bottleneck),
            nn.GELU(),
            nn.BatchNorm1d(num_bottleneck),
            nn.Dropout(p=drop_rate)
        ]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        feature = x
        x = self.classifier(x)
        return x, feature


class eva02mim(nn.Module):
    def __init__(self, class_num, drop_rate, block=4, share_weight=True):
        super(eva02mim, self).__init__()
        self.model_1 = timm.create_model('timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True,
                                         num_classes=0,
                                         features_only=True)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model('timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True,
                                             num_classes=0)
        # self.classifier = ClassBlock(1024, class_num, drop_rate)
        self.block = block
        # self.radialSlicer = RadialSlicer(num_directions=num_directions)

        for i in range(self.block):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(1024, class_num, drop_rate))

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)  # 向下取整
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                # print("x", x.shape) torch.Size([16, 576, 768])
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]
                # print("x_curr", x_curr.shape)
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = functional.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    # print("x_pad", x_pad.size())
                    x_curr = x_curr - x_pad
                # print("x_curr", x_curr.shape)
                avgpool = pooling(x_curr)
                # print("pool", avgpool.shape)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = functional.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = functional.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

    def classifier(self, x):
        part = {}
        predict = {}
        features = []
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1)  # [B, C, H, W] -> [B, C, H*W]
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            # print(c)
            predict[i], feature = c(part[i])
            features.append(feature)
            # print(predict[i].shape)
        # print(predict)
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y, torch.stack(features, dim=2)

    def forward(self, x1, x2):
        if self.training:
            if x1 is None:
                y1 = None
            else:
                x1 = self.model_1(x1)

                x1_lpn = x1[-1]

                x1_lpn = self.get_part_pool(x1_lpn)
                # if self.training:
                #     # 如果在训练阶段，我们可能要通过一个分类器获得类别预测
                y1, f1 = self.classifier(x1_lpn)
                # else:
                #     # 如果在测试阶段，我们可能要返回特征本身
                #     y1 = self.classifier(x1_lpn)

                if x2 is None:
                    y2 = None
                else:
                    x2 = self.model_2(x2)
                    x2_lpn = x2[-1]

                    x2_lpn = self.get_part_pool(x2_lpn)
                    y2, f2 = self.classifier(x2_lpn)
                return y1, y2, f1, f2

        else:
            with torch.no_grad():
                if x1 is None:
                    y1 = None
                else:
                    x1 = self.model_1(x1)

                    x1_lpn = x1[-1]

                    x1_lpn = self.get_part_pool(x1_lpn)
                    y1 = self.classifier(x1_lpn)

                if x2 is None:
                    y2 = None
                else:
                    x2 = self.model_2(x2)
                    x2_lpn = x2[-1]

                    x2_lpn = self.get_part_pool(x2_lpn)
                    y2 = self.classifier(x2_lpn)

            return y1, y2


class eva02_B(nn.Module):
    def __init__(self, class_num, drop_rate, share_weight=True):
        super(eva02_B, self).__init__()
        model = timm.create_model('timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=True,
                                  num_classes=0)
        self.model_1 = model
        if share_weight:
            self.model_2 = model
        else:
            self.model_2 = timm.create_model('timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=True,
                                             num_classes=0)
        self.classifier = ClassBlock(768, class_num, drop_rate)

    def forward(self, x1, x2):
        if self.training:
            if x1 is None:
                y1 = None
                f1 = None
            else:
                x1 = self.model_1(x1)
                y1, f1 = self.classifier(x1)

            if x2 is None:
                y2 = None
                f2 = None
            else:
                x2 = self.model_2(x2)
                y2, f2 = self.classifier(x2)

            return y1, y2, f1, f2
        else:
            with torch.no_grad():
                if x1 is None:
                    y1 = None
                    f1 = None
                else:
                    x1 = self.model_1(x1)
                    y1, f1 = self.classifier(x1)

                if x2 is None:
                    y2 = None
                    f2 = None
                else:
                    x2 = self.model_2(x2)
                    y2, f2 = self.classifier(x2)

            return f1, f2


class eva02mim_s(nn.Module):
    def __init__(self, class_num, drop_rate, block=4, share_weight=True):
        super(eva02mim_s, self).__init__()
        pretrained_cfg = timm.models.create_model('timm/eva02_small_patch14_336.mim_in22k_ft_in1k').default_cfg
        pretrained_cfg[
            'file'] = '/media/sues/daa8aa38-6c2b-4fb6-a66f-327e4fc2f6a6/weights/pre/eva02_mim_s/pytorch_model.bin'
        self.model_1 = timm.create_model('timm/eva02_small_patch14_336.mim_in22k_ft_in1k', pretrained=True,
                                         num_classes=0,
                                         features_only=True, pretrained_cfg=pretrained_cfg)
        if share_weight:
            self.model_2 = self.model_1
        else:
            self.model_2 = timm.create_model('timm/eva02_small_patch14_336.mim_in22k_ft_in1k', pretrained=True,
                                             num_classes=0)
        self.classifier = ClassBlock(384, class_num, drop_rate)
        self.block = block

        for i in range(self.block):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(384, class_num, drop_rate))

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)  # 向下取整
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                # print("x", x.shape) torch.Size([16, 576, 768])
                x_curr = x[:, :, (c_h - i * per_h):(c_h + i * per_h), (c_w - i * per_w):(c_w + i * per_w)]
                # print("x_curr", x_curr.shape)
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = functional.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    # print("x_pad", x_pad.size())
                    x_curr = x_curr - x_pad
                # print("x_curr", x_curr.shape)
                avgpool = pooling(x_curr)
                # print("pool", avgpool.shape)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = functional.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = functional.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

    def classifier(self, x):
        part = {}
        predict = {}
        features = []
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1)  # [B, C, H, W] -> [B, C, H*W]
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier' + str(i)
            c = getattr(self, name)
            # print(c)
            predict[i], feature = c(part[i])
            features.append(feature)
            # print(predict[i].shape)
        # print(predict)
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y, torch.stack(features, dim=2)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            # print(x1[0].shape)
            # print(x1[1].shape)
            # print(x1[2].shape)
            # print(x1[3].shape)
            x1_lpn = x1[-1]
            # x1_end = x1[-1]
            # x1_lpn = self.restore_vit_end_feature(x1_lpn)
            # print(x1_lpn.shape)
            x1_lpn = self.get_part_pool(x1_lpn)
            # print(x1_original.shape)

            # 提取全局和局部特征
            # global_feat = self.global_branch(x1)  # [B, C1, 1, 1]
            # local_feat = self.local_branch(x1)  # [B, C2, 1, 1]
            # x1_end = self.restore_vit_end_feature(x1_end)
            # radialSlicer_feat = self.radialSlicer(x1_end)  # [B, C3, 1, 1]
            # x1_end = F.adaptive_avg_pool2d(x1_end, (4, 1))
            # print(radialSlicer_feat.shape)
            # 沿通道维度拼接特征
            # x1_concat = torch.cat([x1_original, global_feat, local_feat], dim=1)
            # x1_concat = torch.cat([x1_original, radialSlicer_feat, x1_end], dim=1)
            # print(x1_concat.shape)
            if self.training:
                # 如果在训练阶段，我们可能要通过一个分类器获得类别预测
                y1, _ = self.classifier(x1_lpn)
            else:
                # 如果在测试阶段，我们可能要返回特征本身
                y1 = self.classifier(x1_lpn)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            x2_lpn = x2[-1]
            # x2_end = x2[-1]
            # 保留之前的处理步骤
            # x2_lpn = self.restore_vit_end_feature(x2_lpn)
            x2_lpn = self.get_part_pool(x2_lpn)
            # x2_end = self.restore_vit_end_feature(x2_end)
            # 提取全局和局部特征
            # global_feat = self.global_branch(x2)  # [B, C1, 1, 1]
            # radialSlicer_feat = self.radialSlicer(x2_end)  # [B, C3, 1, 1]

            # local_feat = self.local_branch(x2)  # [B, C2, 1, 1]
            # 沿通道维度拼接特征
            # x2_concat = torch.cat([x2_original,  global_feat, local_feat], dim=1)
            # x2_end = F.adaptive_avg_pool2d(x2_end, (4, 1))
            # x2_concat = torch.cat([x2_original, radialSlicer_feat, x2_end], dim=1)

            if self.training:
                # 如果在训练阶段，我们可能要通过一个分类器获得类别预测
                y2, _ = self.classifier(x2_lpn)
            else:
                # 如果在测试阶段，我们可能要返回特征本身
                y2 = self.classifier(x2_lpn)

        return y1, y2


class eva02_S(nn.Module):
    def __init__(self, class_num, drop_rate, share_weight=True):
        super(eva02_S, self).__init__()
        model = timm.create_model('timm/eva02_small_patch14_336.mim_in22k_ft_in1k', pretrained=False,
                                  num_classes=0)
        self.model_1 = model
        if share_weight:
            self.model_2 = model
        else:
            self.model_2 = timm.create_model('timm/eva02_small_patch14_336.mim_in22k_ft_in1k', pretrained=True,
                                             num_classes=0)
        self.classifier = ClassBlock(384, class_num, drop_rate)

    def forward(self, x1, x2):
        if self.training:
            if x1 is None:
                y1 = None
                f1 = None
            else:
                x1 = self.model_1(x1)
                y1, f1 = self.classifier(x1)

            if x2 is None:
                y2 = None
                f2 = None
            else:
                x2 = self.model_2(x2)
                y2, f2 = self.classifier(x2)

            return y1, y2, f1, f2
        else:
            with torch.no_grad():
                if x1 is None:
                    y1 = None
                    f1 = None
                else:
                    x1 = self.model_1(x1)
                    y1, f1 = self.classifier(x1)

                if x2 is None:
                    y2 = None
                    f2 = None
                else:
                    x2 = self.model_2(x2)
                    y2, f2 = self.classifier(x2)

            return f1, f2


class eva02_L(nn.Module):
    def __init__(self, class_num, drop_rate, share_weight=True):
        super(eva02_L, self).__init__()
        model = timm.create_model('timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=False,
                                  num_classes=0)
        self.model_1 = model
        if share_weight:
            self.model_2 = model
        else:
            self.model_2 = timm.create_model('timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True,
                                             num_classes=0)
        self.classifier = ClassBlock(1024, class_num, drop_rate)

    def forward(self, x1, x2):
        if self.training:
            if x1 is None:
                y1 = None
                f1 = None
            else:
                x1 = self.model_1(x1)
                y1, f1 = self.classifier(x1)

            if x2 is None:
                y2 = None
                f2 = None
            else:
                x2 = self.model_2(x2)
                y2, f2 = self.classifier(x2)

            return y1, y2, f1, f2
        else:
            with torch.no_grad():
                if x1 is None:
                    y1 = None
                    f1 = None
                else:
                    x1 = self.model_1(x1)
                    y1, f1 = self.classifier(x1)

                if x2 is None:
                    y2 = None
                    f2 = None
                else:
                    x2 = self.model_2(x2)
                    y2, f2 = self.classifier(x2)

            return f1, f2


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context
    # model = SwinTransformer(class_num=701, drop_rate=0.1, share_weight=True, num_directions=5, block=4).cuda()
    # model = SwinTransformer(class_num=701, drop_rate=0.1, share_weight=True, num_directions=5, block=4).cuda()
    # model = ResNet18(class_num=701, drop_rate=0.1, share_weight=True).cuda()
    # model = swin_exp(class_num=701, drop_rate=0.1, share_weight=False).cuda()
    # model = convnext_lpn(class_num=701, drop_rate=0.1, share_weight=False).cuda()
    model = eva02_S(class_num=701, drop_rate=0.1, share_weight=True).cuda()
    # model = eva02_B(class_num=701, drop_rate=0.1, share_weight=True).cuda()
    # model = ResNet101(class_num=701, drop_rate=0.1, share_weight=True).cuda()
    # model = moblienetv3_large(drop_rate=0.1).cuda()
    # model = SwinTransformer_L(701, 0.1, True).cuda()
    # model = Convnext_L(701, 0.1, True).cuda()
    # model = Convnextv2_huge(701, 0.1, True).cuda()
    # model = SwinTransformer_L(701, 0.1, True).cuda()
    # model = SwinTransformerV2_L(701, 0.1, True).cuda()
    # model = EVA02_L_MIM(701, 0.1, True).cuda()
    # model = FLattenSwinTransformer_B_384(701, 0.1, True).cuda()
    # model = EVA02_B(701, 0.1, True).cuda()
    # model = ft_net_LPN(701, True).cuda()
    # print(model.device)
    # model = ViT(701, 0.1, True).cuda()
    # model = ViT_CGAM(701, 0.1, True).cuda()
    # print(model)
    input1 = torch.randn(8, 3, 336, 336).cuda()
    input2 = torch.randn(8, 3, 336, 336).cuda()
    output = model(input1, input2)
    print(output)


model_dict = {
    "eva02mim_s": eva02mim_s,
    "eva02mim": eva02mim,
    "eva02_L": eva02_L,
    "eva02_B": eva02_B,
    "eva02_S": eva02_S,
}
