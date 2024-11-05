import torch
from torchmetrics import Metric


class KNNCosDistance(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("maxs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, d) -> None:
        max_d, _ = torch.max(d, 0)
        self.maxs += max_d.cpu()
        self.num += 1

    def compute(self) -> torch.Tensor:
        return self.maxs.float() / self.num
