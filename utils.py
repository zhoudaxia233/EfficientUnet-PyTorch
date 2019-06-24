import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


GlobalParams = namedtuple('GlobalParams', ['batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'num_classes',
                                           'width_coefficient', 'depth_coefficient', 'depth_divisor', 'min_depth',
                                           'drop_connect_rate'])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = namedtuple('BlockArgs', ['kernel_size', 'num_repeat', 'input_filters', 'output_filters', 'expand_ratio',
                                     'id_skip', 'strides', 'se_ratio'])
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


IMAGENET_WEIGHTS = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth'
}


def round_filters(filters, global_params):
    """Round number of filters
    """
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of repeats
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def get_efficientnet_params(model_name, override_params=None):
    """Get efficientnet params based on model name
    """
    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        # Note: the resolution here is just for reference, its values won't be used.
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    if model_name not in params_dict.keys():
        raise KeyError('There is no model named {}.'.format(model_name))

    width_coefficient, depth_coefficient, _, dropout_rate = params_dict[model_name]

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=0.2,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None)

    if override_params:
        global_params = global_params._replace(**override_params)

    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params


class BlockDecoder(object):
    """Block Decoder for readability
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=[int(options['s'][0]), int(options['s'][1])]
        )

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
          string_list: a list of strings, each string is a notation of block.
        Returns:
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
          blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
          a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


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


class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions with same padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, groups=groups,
                         bias=bias)
        self.stride = stride if isinstance(stride, (list, tuple)) else [stride] * 2

    def forward(self, x):
        input_h, input_w = x.size()[2:]
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = math.ceil(input_h / stride_h), math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block
    """
    def __init__(self, block_args, global_params):
        super().__init__()

        self.block_args = block_args
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (self.block_args.se_ratio is not None) and (0 < self.block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        self.swish = Swish()

        # Expansion phase
        in_channels = self.block_args.input_filters
        out_channels = self.block_args.input_filters * self.block_args.expand_ratio
        if self.block_args.expand_ratio != 1:
            self.expand_conv = Conv2dSamePadding(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1,
                                                 bias=False)
            self.expand_batch_norm = nn.BatchNorm2d(num_features=out_channels,
                                                    momentum=self.batch_norm_momentum,
                                                    eps=self.batch_norm_epsilon)

        # Depth-wise convolution phase
        kernel_size = self.block_args.kernel_size
        strides = self.block_args.strides
        self.depthwise_conv = Conv2dSamePadding(in_channels=out_channels,
                                                out_channels=out_channels,
                                                groups=out_channels,
                                                kernel_size=kernel_size,
                                                stride=strides,
                                                bias=False)
        self.depthwise_batch_norm = nn.BatchNorm2d(num_features=out_channels,
                                                   momentum=self.batch_norm_momentum,
                                                   eps=self.batch_norm_epsilon)

        # Squeeze and Excitation layer
        if self.has_se:
            num_squeezed_channels = max(1, int(self.block_args.input_filters * self.block_args.se_ratio))
            self.se_reduce = Conv2dSamePadding(in_channels=out_channels,
                                               out_channels=num_squeezed_channels,
                                               kernel_size=1)
            self.se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels,
                                               out_channels=out_channels,
                                               kernel_size=1)

        # Output phase
        final_output_channels = self.block_args.output_filters
        self.project_conv = Conv2dSamePadding(in_channels=out_channels,
                                              out_channels=final_output_channels,
                                              kernel_size=1,
                                              bias=False)
        self.output_batch_norm = nn.BatchNorm2d(num_features=final_output_channels,
                                                momentum=self.batch_norm_momentum,
                                                eps=self.batch_norm_epsilon)

    def forward(self, x, drop_connect_rate=None):
        identity = x
        # Expansion and depth-wise convolution
        if self.block_args.expand_ratio != 1:
            x = self.swish(self.expand_batch_norm(self.expand_conv(x)))
        x = self.swish(self.depthwise_batch_norm(self.depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self.se_expand(self.swish(self.se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self.output_batch_norm(self.project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self.block_args.input_filters, self.block_args.output_filters
        if self.id_skip and self.block_args.strides == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate=drop_connect_rate, training=self.training)
            x = x + identity
        return x
