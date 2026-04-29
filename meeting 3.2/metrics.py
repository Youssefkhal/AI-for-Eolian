import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mse(pred, target))


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denom = torch.mean(target.abs()) + 1e-8
    return rmse(pred, target) / denom


def mape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    denom = target.abs() + 1e-8
    return torch.mean((pred - target).abs() / denom)


def bias_index(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(pred - target) / (torch.mean(target.abs()) + 1e-8)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2) + 1e-8
    return 1.0 - ss_res / ss_tot


def coefficient_of_variation(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    mean = torch.mean(values, dim=dim)
    std = torch.std(values, dim=dim)
    return std / (mean.abs() + 1e-8)
