from efficientunet import *

# This is a demo to show you how to use the library.
if __name__ == '__main__':
    t = torch.rand(2, 3, 224, 224).cuda()

    # EfficientNet test
    model = EfficientNet.from_name('efficientnet-b5', n_classes=5, pretrained=True).cuda()
    print(model(t).size())

    # EfficientUnet test
    b0unet = get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True).cuda()
    print(b0unet(t).size())
