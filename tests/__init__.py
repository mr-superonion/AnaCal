import os
import threading
import time

import psutil
from psutil._common import bytes2human


def print_mem(byin):
    print("Mem:", bytes2human(byin))


def mem_used():
    process = psutil.Process(os.getpid())
    mem_usage_bytes = process.memory_info().rss
    return mem_usage_bytes


def peak_mem_used(func, interval=0.01):
    """Run ``func()`` in this process and return the peak resident set
    size (bytes) seen while it ran, sampled from a daemon thread.  The
    sampler never forks, so it is safe with the C++ thread pool alive.
    """
    process = psutil.Process(os.getpid())
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(interval)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        func()
    finally:
        stop.set()
        sampler.join()
    peak[0] = max(peak[0], process.memory_info().rss)
    return peak[0]
