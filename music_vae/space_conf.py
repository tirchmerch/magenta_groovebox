import numpy as np

space = {
    'free_bits': np.arange(0, 272, step=16),
    'max_beta': np.arange(0.05, 0.55, step=0.05),
    'batch_size': [512]
}