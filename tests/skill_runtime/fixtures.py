import time


def slow_handler(data, _context):
    time.sleep(0.05)
    return data
