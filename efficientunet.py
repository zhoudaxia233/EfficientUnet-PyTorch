from collections import OrderedDict
from layers import *
from efficientnet import EfficientNet


__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = (output.size()[1], output)

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = (output.size()[1], output)

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.out_channels = out_channels
        self.concat_input = concat_input

    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, (n_channels, x) = blocks.popitem()

        x = up_conv(n_channels, 512)(x)
        x = torch.cat([x, blocks.popitem()[1][1]], dim=1)
        x = double_conv(x.size(1), 512)(x)

        x = up_conv(512, 256)(x)
        x = torch.cat([x, blocks.popitem()[1][1]], dim=1)
        x = double_conv(x.size(1), 256)(x)

        x = up_conv(256, 128)(x)
        x = torch.cat([x, blocks.popitem()[1][1]], dim=1)
        x = double_conv(x.size(1), 128)(x)

        x = up_conv(128, 64)(x)
        x = torch.cat([x, blocks.popitem()[1][1]], dim=1)
        x = double_conv(x.size(1), 64)(x)

        if self.concat_input:
            x = up_conv(64, 32)(x)
            x = torch.cat([x, input_], dim=1)
            x = double_conv(x.size(1), 32)(x)

        x = nn.Conv2d(x.size(1), self.out_channels, kernel_size=1)(x)

        return x


def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b1', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b2', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b3', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b5', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b6', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b7', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


# if __name__ == '__main__':
#     t = torch.rand(2, 3, 224, 224)
#     model = get_efficientunet_b5()
#     print(model(t).size())
