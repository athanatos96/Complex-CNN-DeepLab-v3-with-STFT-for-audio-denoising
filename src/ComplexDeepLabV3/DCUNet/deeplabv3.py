import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torchvision.models import _utils



from typing import Optional, Callable, Union, List, Dict, Type#, Any
from collections import OrderedDict


import DCUNet.complex_nn as complex_nn

'''
Reimplementaion of DeepLabV3 to work with complex numbers

DeepLabv3 base code: https://github.com/pytorch/vision/blob/0dceac025615a1c2df6ec1675d8f9d7757432a49/torchvision/models/segmentation/deeplabv3.py
FCN head base code: https://github.com/pytorch/vision/blob/0dceac025615a1c2df6ec1675d8f9d7757432a49/torchvision/models/segmentation/fcn.py#L36
Resnet base code: https://github.com/pytorch/vision/blob/0dceac025615a1c2df6ec1675d8f9d7757432a49/torchvision/models/resnet.py#L166

Author: https://github.com/athanatos96
'''

########################################################################################################
########################################################################################################
######################################## Resnet backbone model #########################################
########################################################################################################
########################################################################################################

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, complex_numbers: Optional[bool] =False):# -> nn.Conv2d:
    """3x3 convolution with padding"""
    #return nn.Conv2d(
    #    in_planes,
    #    out_planes,
    #    kernel_size=3,
    #    stride=stride,
    #    padding=dilation,
    #    groups=groups,
    #    bias=False,
    #    dilation=dilation,
    #)
    if complex_numbers:
        # Using complex numbers requiers to use complex layers
        base_conv = complex_nn.ComplexConv2d
    else:
        # Use normal layers
        base_conv = nn.Conv2d
    
    return base_conv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, complex_numbers: Optional[bool] =False):# -> nn.Conv2d:
    """1x1 convolution"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    if complex_numbers:
        # Using complex numbers requiers to use complex layers
        base_conv = complex_nn.ComplexConv2d
    else:
        # Use normal layers
        base_conv = nn.Conv2d
        
    return base_conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
                self,
                inplanes: int,
                planes: int,
                stride: int = 1,
                downsample: Optional[nn.Module] = None,
                groups: int = 1,
                base_width: int = 64,
                dilation: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                complex_numbers: Optional[bool] =False
            ) -> None:
        
        super().__init__()
        if norm_layer is None:
            #norm_layer = nn.BatchNorm2d
            if complex_numbers:
                norm_layer = complex_nn.ComplexBatchNorm2d
            else:
                norm_layer = nn.BatchNorm2d
                
        if complex_numbers:
            base_relu = complex_nn.ComplexReLU
        else:
            base_relu = nn.ReLU
        
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, complex_numbers=complex_numbers)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, complex_numbers=complex_numbers)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, complex_numbers=complex_numbers)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = base_relu(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Resnet BackBone Model
class ResNet50_backbone(nn.Module):
    def __init__(self, 
                 #block: Type[Union[BasicBlock, Bottleneck]],
                 block: Type[Bottleneck],
                 layers: List[int],
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 input_channels: int =1,
                 complex_numbers: Optional[bool] =False
                ) -> None:
        #print("Resnet50_backbone Init")
        super(ResNet50_backbone, self).__init__()
        
        if complex_numbers:
            # Using complex numbers requiers to use complex layers
            base_conv = complex_nn.ComplexConv2d
            base_bn = complex_nn.ComplexBatchNorm2d
            base_relu = complex_nn.ComplexReLU
            base_maxpool = complex_nn.ComplexMaxPool2d
        else:
            # Use normal layers
            base_conv = nn.Conv2d
            base_bn = nn.BatchNorm2d
            base_relu = nn.ReLU
            base_maxpool = nn.MaxPool2d
            
        
        # Set Normalization layer to batchNorm
        #self._norm_layer = nn.BatchNorm2d
        self._norm_layer = base_bn
        
        self.inplanes = 64
        self.dilation = 1
        self.input_channels = input_channels
        
        # Set dilation
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        
        #self.conv1 = nn.Conv2d(self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(self.inplanes)
        #self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.conv1 = base_conv(self.input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_bn(self.inplanes)
        self.relu = base_relu(inplace=False)
        self.maxpool = base_maxpool(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], complex_numbers=complex_numbers)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], complex_numbers=complex_numbers)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], complex_numbers=complex_numbers)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], complex_numbers=complex_numbers)
    
    def _make_layer(
                    self,
                    #block: Type[Union[BasicBlock, Bottleneck]],
                    block: Type[Bottleneck],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False,
                    complex_numbers: Optional[bool] =False
                ) -> nn.Sequential:
        
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, complex_numbers=complex_numbers),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, complex_numbers=complex_numbers
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    complex_numbers = complex_numbers
                )
            )

        return nn.Sequential(*layers)
        
    def _forward_impl(self, x: Tensor) -> Tensor:
        #print("Input shape", x.shape)
        # See note [TorchScript super()]
        x = self.conv1(x)
        #print("After Conv1 shape", x.shape)
        x = self.bn1(x)
        #print("After bn1 shape", x.shape)
        x = self.relu(x)
        #print("After relu shape", x.shape)
        x = self.maxpool(x)
        #print("After maxpool shape", x.shape)

        x = self.layer1(x)
        #print("After layer1 shape", x.shape)
        x = self.layer2(x)
        #print("After layer2 shape", x.shape)
        x = self.layer3(x)
        #print("After layer3 shape", x.shape)
        
        return_layers = {"aux": x}
        
        
        y = self.layer4(x)
        #print("After layer4 shape", x.shape)
        
        return_layers["out"] = y
        
        return return_layers

    def forward(self, x: Tensor) -> Tensor:
        #print("Forward Function of Resnet50_backbone")
        return self._forward_impl(x)
    
    
    
########################################################################################################
########################################################################################################
########################################      ASPP Layers      #########################################
########################################################################################################
########################################################################################################

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, complex_numbers: Optional[bool] = False) -> None:
        if complex_numbers:
            # Using complex numbers requiers to use complex layers
            base_conv = complex_nn.ComplexConv2d
            base_bn = complex_nn.ComplexBatchNorm2d
            base_relu = complex_nn.ComplexReLU
        else:
            # Use normal layers
            base_conv = nn.Conv2d
            base_bn = nn.BatchNorm2d
            base_relu = nn.ReLU
        
        #modules = [
        #    nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        #    nn.BatchNorm2d(out_channels),
        #    nn.ReLU(),
        #]
        
        modules = [
            base_conv(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            base_bn(out_channels),
            base_relu(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, complex_numbers: Optional[bool] = False) -> None:
        self.complex_numbers = complex_numbers
        if complex_numbers:
            # Using complex numbers requiers to use complex layers
            base_conv = complex_nn.ComplexConv2d
            base_bn = complex_nn.ComplexBatchNorm2d
            base_relu = complex_nn.ComplexReLU
            base_aap = complex_nn.ComplexAdaptiveAvgPool2d
        else:
            # Use normal layers
            base_conv = nn.Conv2d
            base_bn = nn.BatchNorm2d
            base_relu = nn.ReLU
            base_aap = nn.AdaptiveAvgPool2d
            
        modules = [
            base_aap(1),
            base_conv(in_channels, out_channels, 1, bias=False),
            base_bn(out_channels),
            base_relu(),
        ]
        super().__init__(*modules)
        
        #super().__init__(
        #    nn.AdaptiveAvgPool2d(1),
        #    nn.Conv2d(in_channels, out_channels, 1, bias=False),
        #    nn.BatchNorm2d(out_channels),
        #    nn.ReLU(),
        #)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print("ASPPPooling Input shape: ",x.shape)
        #size = x.shape[-2:]
        
        tensor_shape = x.shape
        if(self.complex_numbers):
            # In case complex numbers are separated into a diferent diemsion as float
            size = tensor_shape[-3:-1]
        elif(not self.complex_numbers):
            # in case complex number are together as complex
            size = tensor_shape[-2:]
        #print("ASPPPooling Selected shape: ",size)
        
        for mod in self:
            x = mod(x)
            
        if self.complex_numbers:
            return complex_nn.complex_interpolate(x, size=size, mode="bilinear", align_corners=False)
        else:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            
        #return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256, complex_numbers: Optional[bool] = False) -> None:
        super().__init__()
        
        if complex_numbers:
            # Using complex numbers requiers to use complex layers
            base_conv = complex_nn.ComplexConv2d
            base_bn = complex_nn.ComplexBatchNorm2d
            base_relu = complex_nn.ComplexReLU
            base_dropout = complex_nn.ComplexDropout
        else:
            # Use normal layers
            base_conv = nn.Conv2d
            base_bn = nn.BatchNorm2d
            base_relu = nn.ReLU
            base_dropout = nn.Dropout
        
        
        modules = []
        modules.append(
            #nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
            nn.Sequential(base_conv(in_channels, out_channels, 1, bias=False), base_bn(out_channels), base_relu())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, complex_numbers=complex_numbers))

        modules.append(ASPPPooling(in_channels, out_channels, complex_numbers=complex_numbers))

        self.convs = nn.ModuleList(modules)

        #self.project = nn.Sequential(
        #    nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
        #    nn.BatchNorm2d(out_channels),
        #    nn.ReLU(),
        #    nn.Dropout(0.5),
        #)
        
        self.project = nn.Sequential(
            base_conv(len(self.convs) * out_channels, out_channels, 1, bias=False),
            base_bn(out_channels),
            base_relu(),
            base_dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

########################################################################################################
########################################################################################################
########################################     Deep Lab Head     #########################################
########################################################################################################
########################################################################################################

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, complex_numbers: Optional[bool] = False) -> None:
        
        if complex_numbers:
            # Using complex numbers requiers to use complex layers
            base_conv = complex_nn.ComplexConv2d
            base_bn = complex_nn.ComplexBatchNorm2d
            base_relu = complex_nn.ComplexReLU
        else:
            # Use normal layers
            base_conv = nn.Conv2d
            base_bn = nn.BatchNorm2d
            base_relu = nn.ReLU
        
        layers = [
            ASPP(in_channels, [12, 24, 36], complex_numbers=complex_numbers),
            base_conv(256, 256, 3, padding=1, bias=False),
            base_bn(256),
            base_relu(),
            base_conv(256, num_classes, 1),
        ]
        super().__init__(*layers)
        
        #super().__init__(
        #    ASPP(in_channels, [12, 24, 36]),
        #    nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #    nn.BatchNorm2d(256),
        #    nn.ReLU(),
        #    nn.Conv2d(256, num_classes, 1),
        #)


########################################################################################################
########################################################################################################
########################################        FCN Head        ########################################
########################################################################################################
########################################################################################################

# FCN head
class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int, complex_numbers: Optional[bool] = False) -> None:
        inter_channels = in_channels // 4
        
        if complex_numbers:
            # Using complex numbers requiers to use complex layers
            base_conv = complex_nn.ComplexConv2d
            base_bn = complex_nn.ComplexBatchNorm2d
            base_relu = complex_nn.ComplexReLU
            base_dropout = complex_nn.ComplexDropout
        else:
            # Use normal layers
            base_conv = nn.Conv2d
            base_bn = nn.BatchNorm2d
            base_relu = nn.ReLU
            base_dropout = nn.Dropout
        
        
        #layers = [
        #    nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #    nn.BatchNorm2d(inter_channels),
        #    nn.ReLU(),
        #    nn.Dropout(0.1),
        #    nn.Conv2d(inter_channels, channels, 1),
        #]
        
        layers = [
            base_conv(in_channels, inter_channels, 3, padding=1, bias=False),
            base_bn(inter_channels),
            base_relu(),
            base_dropout(0.1),
            base_conv(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


########################################################################################################
########################################################################################################
####################################### Deeplabv3_resnet50 model #######################################
########################################################################################################
########################################################################################################

class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None, complex_numbers: Optional[bool] = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        
        self.complex_numbers = complex_numbers

    def forward(self, bd) -> Dict[str, Tensor]:
        if self.complex_numbers:
            x = bd['X']
        else:
            x = bd['mag_X']
            
        #input_shape = x.shape[-2:]
        #print("_SimpleSegmentationModel Input shape: ",x.shape)
        tensor_shape = x.shape
        if(len(tensor_shape)==5 and tensor_shape[-1]==2):
            # In case complex numbers are separated into a diferent diemsion as float
            input_shape = tensor_shape[-3:-1]
        elif(len(tensor_shape)==4):
            # in case complex number are together as complex
            input_shape = tensor_shape[-2:]
        #print("_SimpleSegmentationModel Selected shape: ",input_shape)
        
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        
        if self.complex_numbers:
            x = complex_nn.complex_interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            if self.complex_numbers:
                x = complex_nn.complex_interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            else:
                x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        
        #mask = torch.tanh(result["out"])
        #bd['M_hat'] = mask
        #return bd
        
        # Create the Complex Mask Using Polar Coordinates
        #print("Shape of tensor at the end of DeepLab v3:", result["out"].shape)
        if self.complex_numbers:
            unbounded_magnitude = torch.sqrt(torch.sum(torch.pow(result["out"], 2), dim=-1))
            bounded_magnitude = torch.tanh(unbounded_magnitude)
            unit_real = torch.div(result["out"][..., 0], unbounded_magnitude)
            unit_imag = torch.div(result["out"][..., 1], unbounded_magnitude)
            mask_real = torch.mul(bounded_magnitude, unit_real)
            mask_imag = torch.mul(bounded_magnitude, unit_imag)
            mask=torch.stack((mask_real, mask_imag), dim=-1)
        else:
            mask = torch.tanh(result["out"])
        #print("Shape of mask at the end of DeepLab v3:", mask.shape)
        
        bd['M_hat'] = mask
        return bd
        #return result
    
class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass

def _deeplabv3_resnet(
                        backbone: ResNet50_backbone,
                        num_classes: int,
                        aux: Optional[bool],
                        complex_numbers: Optional[bool] = False,
                     ) -> DeepLabV3:
    
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    
    #backbone = _utils.IntermediateLayerGetter(backbone, return_layers=return_layers)

    
    aux_classifier = FCNHead(1024, num_classes, complex_numbers=complex_numbers) if aux else None
    classifier = DeepLabHead(2048, num_classes, complex_numbers=complex_numbers)
    return DeepLabV3(backbone, classifier, aux_classifier, complex_numbers=complex_numbers)



def deeplabv3_resnet50(
            input_channels: int = 1,
            progress: bool = True,
            num_classes: Optional[int] = None,
            aux_loss: Optional[bool] = None,
            complex_numbers: Optional[bool] =False,
        ) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    .. betastatus:: segmentation module
    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.
    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused
    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet50_Weights
        :members:
    """
    if num_classes is None:
        num_classes = 1
    if aux_loss is None:
        aux_loss = True
    if input_channels is None:
        input_channels = 1

    #backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
    #backbone = resnet50(weights = None, replace_stride_with_dilation = [False, True, True])
    backbone = ResNet50_backbone(block = Bottleneck,
                                 layers = [3, 4, 6, 3], 
                                 replace_stride_with_dilation=[False, True, True],
                                 input_channels=input_channels,
                                 complex_numbers=complex_numbers)
    
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss, complex_numbers=complex_numbers)


    return model