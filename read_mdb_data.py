"""
author: Wenyu Qiu
date: 20220330
"""
import os
from torch.utils.data import DataLoader
import config
import torch
import warnings
import importlib
import string
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

data = importlib.import_module('data')
utils = importlib.import_module('utils')
TransLabelConverter = utils.TransLabelConverter
hierarchical_dataset = data.hierarchical_dataset
AlignCollate = data.AlignCollate
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Config(config.Config):
    valid_data = os.path.join(config.data_dir, 'validation')
    workers = 0
    batch_size = 32
    num_class = 40
    with_bilstm = True
    sensitive = False
    filter_punctuation = False
    backbone = 'resnet'

    checkpoint_dir = config.checkpoint_dir
    saved_model = ''

cfg = Config()

if __name__ == '__main__':
    cfg.sensitive = True if 'sensitive' in cfg.saved_model else False
    AlignCollate_valid = AlignCollate()
    
    if cfg.sensitive:
        cfg.character = string.digits + string.ascii_letters + cfg.punctuation
    
    # ONLY need to input the path here, and then the data inside will be read, through the following code
    cfg.valid_data = os.path.join(config.data_dir, 'evaluation', 'CUTE80')

    converter = TransLabelConverter(cfg.character, device)
    cfg.num_class = len(converter.character)
    
    valid_dataset = hierarchical_dataset(cfg.valid_data, cfg.imgH, cfg.imgW, cfg.batch_max_length, cfg.character,
                                         cfg.sensitive, cfg.rgb, cfg.data_filtering_off)
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=int(cfg.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    
    length_of_data = 0
    
    for i, (image_tensors, labels) in enumerate(valid_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        one_image = image[0, :, :, :]
        one_img = one_image.reshape((one_image.shape[1], one_image.shape[2]))
    
    print('length_of_data:{}'.format(length_of_data))