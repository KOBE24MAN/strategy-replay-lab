import random


def synthetic_returns(days: int = 360, seed: int = 19) -> list[float]:
    rng = random.Random(seed)
    values = []
    previous = 0.0
    for day in range(days):
        drift = 0.0012 if (day // 30) % 2 == 0 else -0.0010
        previous = drift + 0.25 * previous + rng.gauss(0, 0.004)
        values.append(previous)
    return values
