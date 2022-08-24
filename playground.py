import torch

if __name__ == '__main__':
    test = torch.randn((4, 5, 6))

    print(test)

    print()

    print(test.argmax(dim=-1))