from efficientnet import *

# This is a demo to show you how to use the library.
if __name__ == '__main__':
    t = torch.rand(2, 3, 224, 224)

    # encoder test
    for i in range(8):
        print(f'efficientnet-b{i}')
        encoder = EfficientNet.encoder(f'efficientnet-b{i}', pretrained=True)
        print(encoder(t).size())
        print()

    # model test
    model = EfficientNet.from_name('efficientnet-b7', n_classes=5, pretrained=True)
    print(model(t).size())
