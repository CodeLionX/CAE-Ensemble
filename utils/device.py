import os

import numpy as np
import torch


def get_free_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        os.system(
            "nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >/tmp/gpu-free-memory.txt"
        )
        memory_available = [
            int(x.split()[2])
            for x in open("/tmp/gpu-free-memory.txt", "r+").readlines()
        ]
        id = np.argmax(memory_available)
        device = device + ":" + str(id)
    return device
