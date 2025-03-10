import csv
import shutil
from pathlib import Path
from typing import List

import torch
from torch import nn, optim


def ckpt_save(
		path: Path,
		model: nn.Module,
		optimizer: optim.Optimizer,
):
	ckpt = {
		"model": model.state_dict(),
		"optim": optimizer.state_dict(),
	}
	torch.save(ckpt, path)


def log_save(path: Path, log: List):
	with path.open("w") as file:
		header = log[0].keys()
		writer = csv.DictWriter(file, fieldnames=header, delimiter="\t")
		writer.writeheader()
		writer.writerows(log)


def cfg_save(path: Path, cfg_paths: List[Path]):
	for cfg_path in cfg_paths:
		shutil.copy(cfg_path, path)