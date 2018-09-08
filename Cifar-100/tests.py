import torch
import numpy as np

# y = torch.zeros(256, 10)
# index = torch.ones(256).long()
# y.gather(1, index.view(-1, 1)) = 1
# y = torch.zeros(256, 10)
y = np.zeros((256, 10))
index = np.ones(256).astype('int32')
y[:, index] = 1
print y

