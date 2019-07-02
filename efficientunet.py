from collections import OrderedDict
from layers import *


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
#
#
# if __name__ == '__main__':
#     from efficientnet import *
#
#     t = torch.rand(2, 3, 224, 224)
#     for i in range(8):
#         encoder = EfficientNet.encoder(f'efficientnet-b{i}', pretrained=True)
#         model = EfficientUnet(encoder)
#         print(model(t).size())
