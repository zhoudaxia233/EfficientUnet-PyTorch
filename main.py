from efficient import *

# This is a demo to show you how to use the library.
if __name__ == '__main__':
    model = EfficientNet.from_name('efficientnet-b7', n_classes=5, pretrained=True)
    x = torch.rand(2, 3, 224, 224)
    print(model(x).size())
