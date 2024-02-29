import torch

def check_cuda():
    
    return torch.cuda.is_available()


check_cuda()

