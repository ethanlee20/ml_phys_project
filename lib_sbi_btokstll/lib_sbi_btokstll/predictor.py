
import torch
from torch.nn.functional import log_softmax


class Predictor:

    def __init__(self, model, features, device):

        if not isinstance(features, torch.Tensor):
            raise ValueError("Features must be torch tensor.")

        self.device = device
        self.model = model.to(device)
        self.features = features.to(device)

    def calc_log_probs(self):

        with torch.no_grad():
            event_logits = self.model(self.features)
            event_log_probs = log_softmax(event_logits, dim=2)
            set_logits = torch.sum(event_log_probs, dim=1)
            set_log_probs = log_softmax(set_logits, dim=1)
            return set_log_probs
    
    


