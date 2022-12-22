import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Original code: https://github.com/sweetcocoa/DeepComplexUNetPyTorch/blob/master/DCUNet/complex_nn.py

New functions adapted from https://github.com/wavefrontshaping/complexPyTorch/blob/70a511c1bedc4c7eeba0d571638b35ff0d8347a2/complexPyTorch/complexFunctions.py
They were built to run with complex types for pytorch. 
I had to change them to work with floats with 1 extra dimension of size 2 (Real, Imaginary)

New Functions and classes:
ComplexAdaptiveAvgPool2d
ComplexMaxPool2d
ComplexReLU
ComplexDropout
complex_interpolate

By: https://github.com/athanatos96

'''
# Original code: https://github.com/sweetcocoa/DeepComplexUNetPyTorch/blob/master/DCUNet/complex_nn.py
# New functions based on https://github.com/wavefrontshaping/complexPyTorch/blob/70a511c1bedc4c7eeba0d571638b35ff0d8347a2/complexPyTorch/complexFunctions.py


# Need for max pool
def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(dim=-1, index=indices.flatten(start_dim=-2)).view_as(indices)
    return output


class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output

class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, out_size, **kwargs):
        super().__init__()
        self.aap_re = nn.AdaptiveAvgPool2d(out_size, **kwargs)
        self.aap_im = nn.AdaptiveAvgPool2d(out_size, **kwargs)
        
    def forward(self, x):
        real = self.aap_re(x[..., 0])
        imag = self.aap_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output

class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, **kwargs):
        super().__init__()
        self.com_maxpool =  nn.MaxPool2d(kernel_size=kernel_size, 
                                                stride=stride, 
                                                padding=padding, 
                                                dilation=dilation, 
                                                return_indices=True, 
                                                ceil_mode=ceil_mode, **kwargs)
    def forward(self, x):
        # calculate abs value
        x_abs = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1))
        x_angle = torch.atan2(x[..., 1], x[..., 0])
        
        '''
        Perform complex max pooling by selecting on the absolute value on the complex values.
        '''
        absolute_value, indices = self.com_maxpool(x_abs)
        
        # get only the phase values selected by max pool
        x_angle = _retrieve_elements_from_indices(x_angle, indices)


        real = absolute_value*torch.cos(x_angle)
        imag = absolute_value*torch.sin(x_angle)
        output = torch.stack((real, imag), dim=-1)
        return output

class ComplexReLU(nn.Module):    
    def __init__(self, inplace=False, **kwargs):
        super().__init__()
        
        self.relu_re = nn.ReLU(inplace=inplace, **kwargs)
        self.relu_im = nn.ReLU(inplace=inplace, **kwargs)
        
    def forward(self, x):
        real = self.relu_re(x[..., 0])
        imag = self.relu_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output
    
class ComplexDropout(nn.Module):    
    def __init__(self, p=0.5, inplace=False, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        
        self.p = p
        
    def forward(self, x):
        
        device = x.device
        mask = torch.ones(*x.shape[:-1], dtype = torch.float32, device = device)
        mask = self.dropout(mask)*1/(1-self.p)
        mask = mask.type(x.dtype)
        
        real = mask*x[..., 0]
        imag = mask*x[..., 1]
        output = torch.stack((real, imag), dim=-1)
        return output

def complex_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    real = input[..., 0]
    imag = input[..., 1]
    
    real = F.interpolate(real, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    imag = F.interpolate(imag, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    
    output = torch.stack((real, imag), dim=-1)
    return output