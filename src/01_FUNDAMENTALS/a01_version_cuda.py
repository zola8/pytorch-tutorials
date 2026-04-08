import subprocess

import torch

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())

    print("\n\n")

    subprocess.Popen('nvidia-smi', shell=True)
