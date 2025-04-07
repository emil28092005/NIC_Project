import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np

t = torch.tensor([1, 2, 3]).float()
print(torch.argmax(t))