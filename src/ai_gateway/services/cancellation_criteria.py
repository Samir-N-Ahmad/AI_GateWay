import torch
from transformers import StoppingCriteria
from threading import Event

class CancellationCriteria(StoppingCriteria):
    def __init__(self, cancel_event: Event):
        self.cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full(
            (input_ids.shape[0],),
            self.cancel_event.is_set(),
            dtype=torch.bool,
            device=input_ids.device
        )