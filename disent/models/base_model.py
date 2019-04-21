import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        raise NotImplementedError
    
