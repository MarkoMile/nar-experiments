import torch

def stack_hidden(input_hidden, hidden, last_hidden, use_last_hidden):
    if use_last_hidden:
        return torch.cat([input_hidden, hidden, last_hidden], dim=-1)
    else:
        return torch.cat([input_hidden, hidden], dim=-1)
    
class NaNException(Exception):
    pass