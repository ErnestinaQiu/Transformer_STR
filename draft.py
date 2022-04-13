# import string

# punctuation = r"""'.-"""
# a = string.digits + string.ascii_letters + punctuation
# print(a)

# import torch
# print('--- {} ---'.format(0))
# print(torch.cuda.is_available())
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('--- {} ---'.format(1))
# print(device)
# print('--- {} ---'.format(2))
# print(torch.cuda.current_device())
# print('--- {} ---'.format(3))
# print(torch.cuda.device_count())
# print('--- {} ---'.format(4))
# print(torch.cuda.get_device_capability(device=0))

import torch

dtype = torch.cuda.FloatTensor
