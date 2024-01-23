import random
import numpy as np
import torch


class FixedSeed:
    def __init__(self, seed):
        self.seed = seed
        self.np_rng_state = None
        self.rand_rng_state = None
        self.pt_rng_state = None
        self.cuda_rng_state = None

    def __enter__(self):
        self.rand_rng_state = random.getstate()
        self.np_rng_state = np.random.get_state()
        self.pt_rng_state = torch.random.get_rng_state()
        self.cuda_rng_state = torch.cuda.get_rng_state_all()

        self.seed_all(seed=self.seed)

    def __exit__(self, *_):
        random.setstate(self.rand_rng_state)
        np.random.set_state(self.np_rng_state)
        torch.random.set_rng_state(self.pt_rng_state)
        torch.cuda.set_rng_state_all(self.cuda_rng_state)

    @staticmethod
    def seed_all(seed=None):
        if isinstance(seed, int) and seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
