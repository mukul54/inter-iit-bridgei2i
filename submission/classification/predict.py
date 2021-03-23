import torch
print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1)))
print(torch.backends.cudnn.version())