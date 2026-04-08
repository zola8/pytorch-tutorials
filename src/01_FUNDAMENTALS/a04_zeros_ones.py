import torch

if __name__ == '__main__':
    print("Zeros:")
    zeros = torch.zeros(size=(3, 4))
    print(zeros)
    # for masking

    print("\nRandom * Zeros:")
    random_tensor = torch.rand(3, 4)
    print(random_tensor * zeros)

    print("\nOnes:")
    ones = torch.ones(size=(3, 4))
    print(ones.dtype)
