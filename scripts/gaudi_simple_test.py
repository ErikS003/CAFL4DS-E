"""Simple test for Gaudi compatibility."""

import torch  # deptry: ignore[DEP003, DEP001]
from loguru import logger

device = torch.device("hpu")
logger.info(f"Detected: {torch.hpu.device_count()}x {device}")
a = torch.tensor(1, device=device)
logger.info(f"Tensor on HPU: {a}")
