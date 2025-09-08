import torch
import MCRViT
if __name__ == '__main__':
    input_tensor = torch.randn(64, 3, 128, 128)
    model = MCRViT.mk_MCRViT(num_classes=10)
    output = model(input_tensor)
    print(output)


