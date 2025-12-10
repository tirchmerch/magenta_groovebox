import numpy as np

space = {
    'free_bits': np.arrange(0, 256, step=16),
    'max_beta': np.arange(0.05, 0.5, step=0.05),
    'batch_size': 512
}