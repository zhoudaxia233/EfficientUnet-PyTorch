import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, name=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups,
                         bias=bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.name = name

    def forward(self, x):
        input_h, input_w = x.size()[2:]
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = math.ceil(input_h / stride_h), math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, name=None):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine,
                         track_running_stats=track_running_stats)
        self.name = name


def drop_connect(inputs, drop_connect_rate, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - drop_connect_rate
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block
    """

    def __init__(self, block_args, global_params, idx):
        super().__init__()

        block_name = 'blocks_' + str(idx) + '_'

        self.block_args = block_args
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        self.swish = Swish(block_name + '_swish')

        # Expansion phase
        in_channels = self.block_args.input_filters
        out_channels = self.block_args.input_filters * self.block_args.expand_ratio
        if self.block_args.expand_ratio != 1:
            self._expand_conv = Conv2dSamePadding(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=1,
                                                  bias=False,
                                                  name=block_name + 'expansion_conv')
            self._bn0 = BatchNorm2d(num_features=out_channels,
                                    momentum=self.batch_norm_momentum,
                                    eps=self.batch_norm_epsilon,
                                    name=block_name + 'expansion_batch_norm')

        # Depth-wise convolution phase
        kernel_size = self.block_args.kernel_size
        strides = self.block_args.strides
        self._depthwise_conv = Conv2dSamePadding(in_channels=out_channels,
                                                 out_channels=out_channels,
                                                 groups=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=strides,
                                                 bias=False,
                                                 name=block_name + 'depthwise_conv')
        self._bn1 = BatchNorm2d(num_features=out_channels,
                                momentum=self.batch_norm_momentum,
                                eps=self.batch_norm_epsilon,
                                name=block_name + 'depthwise_batch_norm')

        # Squeeze and Excitation layer
        if self.has_se:
            num_squeezed_channels = max(1, int(self.block_args.input_filters * self.block_args.se_ratio))
            self._se_reduce = Conv2dSamePadding(in_channels=out_channels,
                                                out_channels=num_squeezed_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_reduce')
            self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                name=block_name + 'se_expand')

        # Output phase
        final_output_channels = self.block_args.output_filters
        self._project_conv = Conv2dSamePadding(in_channels=out_channels,
                                               out_channels=final_output_channels,
                                               kernel_size=1,
                                               bias=False,
                                               name=block_name + 'output_conv')
        self._bn2 = BatchNorm2d(num_features=final_output_channels,
                                momentum=self.batch_norm_momentum,
                                eps=self.batch_norm_epsilon,
                                name=block_name + 'output_batch_norm')

    def forward(self, x, drop_connect_rate=None):
        identity = x
        # Expansion and depth-wise convolution
        if self.block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self.swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self.block_args.input_filters, self.block_args.output_filters
        if self.id_skip and self.block_args.strides == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate=drop_connect_rate, training=self.training)
            x = x + identity
        return x


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


def custom_head(in_channels, out_channels):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_channels, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, out_channels)
    )
