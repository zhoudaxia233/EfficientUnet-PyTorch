from efficientnet import *

# This is a demo to show you how to use the library.
if __name__ == '__main__':
    t = torch.rand(2, 3, 224, 224)

    # encoder test
    encoder = EfficientNet.encoder('efficientnet-b4', pretrained=True)
    print(encoder(t).size())

    # model test
    model = EfficientNet.from_name('efficientnet-b7', n_classes=5, pretrained=True)
    print(model(t).size())
