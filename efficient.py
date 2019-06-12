from torch.hub import load_state_dict_from_url
from utils import *


class EfficientNet(nn.Module):

    def __init__(self, block_args_list, global_params):
        super().__init__()

        self.block_args_list = block_args_list
        self.global_params = global_params

        # Batch norm parameters
        batch_norm_momentum = 1 - self.global_params.batch_norm_momentum
        batch_norm_epsilon = self.global_params.batch_norm_epsilon

        # Stem
        in_channels = 3
        out_channels = round_filters(32, self.global_params)
        self.stem_conv = Conv2dSamePadding(in_channels,
                                           out_channels,
                                           kernel_size=3,
                                           stride=2,
                                           bias=False)
        self.stem_batch_norm = nn.BatchNorm2d(num_features=out_channels,
                                              momentum=batch_norm_momentum,
                                              eps=batch_norm_epsilon)

        # Build blocks
        self.blocks = nn.ModuleList([])
        for block_args in self.block_args_list:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self.global_params),
                output_filters=round_filters(block_args.output_filters, self.global_params),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self.blocks.append(MBConvBlock(block_args, self.global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])

            # The rest of the blocks
            for _ in range(block_args.num_repeat - 1):
                self.blocks.append(MBConvBlock(block_args, self.global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self.global_params)
        self.head_conv = Conv2dSamePadding(in_channels,
                                           out_channels,
                                           kernel_size=1,
                                           bias=False)
        self.head_batch_norm = nn.BatchNorm2d(num_features=out_channels,
                                              momentum=batch_norm_momentum,
                                              eps=batch_norm_epsilon)

        # Final linear layer
        self.dropout_rate = self.global_params.dropout_rate
        self.fc = nn.Linear(out_channels, self.global_params.num_classes)

    def __feature_extractor(self, x):
        # Stem
        x = self.stem_conv(x)
        x = self.stem_batch_norm(x)
        x = swish(x)

        # Blocks
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self.blocks)
            x = block(x, drop_connect_rate)

        # Head
        x = self.head_conv(x)
        x = self.head_batch_norm(x)
        x = swish(x)

        return x

    def forward(self, x):
        x = self.__feature_extractor(x)

        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, *, n_classes=1000, pretrained=False):
        return _get_model_by_name(model_name, classes=n_classes, pretrained=pretrained)

    @classmethod
    def encoder(cls, model_name, *, pretrained=False):
        model = cls.from_name(model_name, pretrained=pretrained)
        return model.__feature_extractor


def _get_model_by_name(model_name, classes=1000, pretrained=False):
    block_args_list, global_params = get_efficientnet_params(model_name, override_params={'num_classes': classes})
    model = EfficientNet(block_args_list, global_params)
    try:
        if pretrained:
            state_dict = load_state_dict_from_url(IMAGENET_WEIGHTS[model_name])
            model.load_state_dict(state_dict)
    except KeyError as e:
        print(f"NOTE: Currently model {e} doesn't have pretrained weights, therefore a model with randomly initialized"
              " weights is returned.")

    return model
