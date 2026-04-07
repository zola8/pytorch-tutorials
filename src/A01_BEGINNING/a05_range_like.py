import torch

if __name__ == '__main__':
    print(torch.arange(0, 11))
    print(torch.arange(-5.0, 5.0))
    print(torch.arange(start=0, end=11, step=2))

    one_to_ten = torch.arange(start=0, end=11)
    ten_zeroes = torch.zeros_like(one_to_ten)
    print(ten_zeroes)
