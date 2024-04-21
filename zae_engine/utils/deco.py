import time


def tictoc(func):
    def wrapper(*args, **kwargs):
        kickoff = time.time()
        out = func(*args, **kwargs)
        print(f"Elapsed time [sec]: {time.time() - kickoff}")
        return out

    return wrapper
