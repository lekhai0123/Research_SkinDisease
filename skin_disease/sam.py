import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.rho = rho
        self.adaptive = adaptive

    @torch.no_grad()
    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        norms = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group.get("adaptive", self.adaptive):
                    norms.append((torch.abs(p) * p.grad).norm(p=2).to(device))
                else:
                    norms.append(p.grad.norm(p=2).to(device))

        if len(norms) == 0:
            return torch.tensor(0.0, device=device)

        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        if grad_norm.item() == 0.0:
            if zero_grad:
                self.zero_grad(set_to_none=True)
            return

        for group in self.param_groups:
            scale = group.get("rho", self.rho) / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                if group.get("adaptive", self.adaptive):
                    e_w = torch.pow(p, 2) * p.grad * scale
                else:
                    e_w = p.grad * scale

                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.state[p].get("e_w", None)
                if e_w is not None:
                    p.sub_(e_w)

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("Use first_step and second_step with SAM.")

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
