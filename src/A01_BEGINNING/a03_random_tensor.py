import torch

if __name__ == '__main__':
    # why is it important?

    # Random numbers are important as many NN (Neural Network) start with tensors with random numbers,
    # and then adjust those to better represent the data.

    random_tensor = torch.rand(2, 3, 4)

    print(random_tensor)
    print(random_tensor.shape)

    random_image_size_tensor = torch.rand(size=(3, 224, 224))
    # color channels (RGB), height, width
    print(random_image_size_tensor.shape, random_image_size_tensor.ndim)
