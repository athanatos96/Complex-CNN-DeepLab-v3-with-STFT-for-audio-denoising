# Complex Deep-Lab V3

Paper: [Phase-Aware Speech Enhancement with Deep Complex U-Net](https://openreview.net/forum?id=SkeRTsAcYm), (A. C. Parra, 2022) 


PyTorch Implementation of [Phase-Aware Speech Enhancement with Deep Complex U-Net](https://openreview.net/forum?id=SkeRTsAcYm), (A. C. Parra, 2022) 


## Original Code

Original Code from https://github.com/sweetcocoa/DeepComplexUNetPyTorch/


Code was adapted to work for Deep Lab V3 [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587), (L-C. Chen et al., 2017) 

New functions adapted from https://github.com/wavefrontshaping/complexPyTorch/blob/70a511c1bedc4c7eeba0d571638b35ff0d8347a2/complexPyTorch/complexFunctions.py
They were built to run with complex types for pytorch. 
I had to change them to work with floats with 1 extra dimension of size 2 (Real, Imaginary)

New Functions and classes:
ComplexAdaptiveAvgPool2d
ComplexMaxPool2d
ComplexReLU
ComplexDropout
complex_interpolate


Reimplementaion of DeepLabV3 to work with complex numbers
DeepLabv3 base code: https://github.com/pytorch/vision/blob/0dceac025615a1c2df6ec1675d8f9d7757432a49/torchvision/models/segmentation/deeplabv3.py
FCN head base code: https://github.com/pytorch/vision/blob/0dceac025615a1c2df6ec1675d8f9d7757432a49/torchvision/models/segmentation/fcn.py#L36
Resnet base code: https://github.com/pytorch/vision/blob/0dceac025615a1c2df6ec1675d8f9d7757432a49/torchvision/models/resnet.py#L166



## Requirements
```text
torch==1.1
soundfile==0.9.0
easydict==1.9
git+https://github.com/keunwoochoi/torchaudio-contrib@61fc6a804c941dec3cf8a06478704d19fc5e415a
git+https://github.com/sweetcocoa/PinkBlack@e45a65623c1b511181f7ea697ca841a7b2900f17
torchcontrib==0.0.2
git+https://github.com/vBaiCai/python-pesq
# gputil # if you need to execute multiple training process
```

## Train
---
1. Download Datasets:
- [https://datashare.is.ed.ac.uk/handle/10283/2791](https://datashare.is.ed.ac.uk/handle/10283/2791)

4. Train
```bash
python ComplexUNet_code/DeepComplexUNetPyTorch-master/train_dcunet.py \
					--batch_size 2 \
					--train_signal Data/DS_10283_2791/Train/clean_trainset_28spk_wav \
					--train_noise Data/DS_10283_2791/Train/noisy_trainset_28spk_wav \
					--test_signal Data/DS_10283_2791/Test/clean_testset_wav \
					--test_noise Data/DS_10283_2791/Test/noisy_testset_wav \
					--ckpt checkpoints/checkpoint.pth \
					--num_step 300 \
					--validation_interval 150\
					--complex



# You can check other arguments from the source code. ( Sorry for the lack description. )                        
```