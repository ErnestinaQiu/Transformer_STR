"""
date: 20220412
author: Wenyu Qiu
des: Because of some issues caused by packages importing, create this to run the same content 
in tools/train.py
"""
from tools.train import *


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

    cfg = Config()

    random.seed(cfg.manualSeed)
    np.random.seed(cfg.manualSeed)
    torch.manual_seed(cfg.manualSeed)
    torch.cuda.manual_seed(cfg.manualSeed)
    
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    train(cfg)
    
    