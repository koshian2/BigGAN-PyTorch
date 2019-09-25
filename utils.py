import torch
import json

# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class EMA(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay 
                                                + self.source_dict[key].data * (1 - decay))


# BigGAN version of othogonal regularization
# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def orthogonal_regularization(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Only apply this to parameters with at least 2 axes, and not in the blacklist
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            w = param.view(param.shape[0], -1)
            grad = (2 * torch.mm(torch.mm(w, w.t()) 
                    * (1. - torch.eye(w.shape[0], device=w.device)), w))  # 行列微分の公式
            # refrence : https://qiita.com/AnchorBlues/items/8fe2483a3a72676eb96d
            param.grad.data += strength * grad.view(param.shape)

# hinge loss
class HingeLoss():
    def __init__(self, batch_size, device):
        self.ones = torch.ones(batch_size, 1).to(device)
        self.zeros = torch.zeros(batch_size, 1).to(device)

    def __call__(self, logits, loss_type):
        assert loss_type in ["gen", "dis_real", "dis_fake"]
        batch_len = len(logits)
        if loss_type == "gen":
            return -torch.mean(logits)
        elif loss_type == "dis_real":
            minval = torch.min(logits - 1, self.zeros[:batch_len])
            return -torch.mean(minval)
        else:
            minval = torch.min(-logits - 1, self.zeros[:batch_len])
            return - torch.mean(minval)

def save_model(model, output_path, use_multi_gpu):
    if use_multi_gpu:
        torch.save(model.module.state_dict(), output_path)
    else:
        torch.save(model.state_dict(), output_path)


def load_settings(parser_obj, filepath, cases_idx):
    with open(filepath, "r") as fp:
        data = json.load(fp)[cases_idx]
    for key, value in data.items():
        if parser_obj.__contains__(key):
            parser_obj.__setattr__(key, value)
    return parser_obj

