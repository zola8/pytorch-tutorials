import torch

if __name__ == '__main__':
    # scalar
    scalar = torch.tensor(7)
    print("scalar:", scalar)
    print("dimension:", scalar.ndim)
    print("value:", scalar.item())

    print()

    # vector
    vector = torch.tensor([7, 7])
    print("vector:", vector)
    print("dimension:", vector.ndim)
    print("shape:", vector.shape)

    print()

    # matrix
    matrix1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("matrix:", matrix1)
    print("matrix (1st dim):", matrix1[1])
    print("matrix (element):", matrix1[1][2])

    print("dimension:", matrix1.ndim)
    print("shape:", matrix1.shape)

    print()

    # tensor
    tensor1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    print("dimension:", tensor1.ndim)
    print("shape:", tensor1.shape)
