import os

import psutil
from psutil._common import bytes2human

from . import sim


def print_mem(byin):
    print("Mem:", bytes2human(byin))


def mem_used():
    process = psutil.Process(os.getpid())
    mem_usage_bytes = process.memory_info().rss
    return mem_usage_bytes
