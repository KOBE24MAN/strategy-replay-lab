from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import random

from market_fixture import synthetic_returns


State = tuple[int, int]
Policy = Callable[[State], int]
FEE_BPS = 5
SLIPPAGE_BPS = 3


@dataclass(frozen=True)
class ReplayResult:
    net_return: float
    max_drawdown: float
    traded_units: int


def market_state(observed_return: float, held: int) -> State:
    return (1 if observed_return > 0.001 else -1 if observed_return < -0.001 else 0, held)


def transition(held: int, target: int, next_return: float, cost: float) -> float:
    return target * next_return - abs(target - held) * cost


def train_policy(returns: list[float], cost: float, *, seed: int = 23, epochs: int = 60) -> Policy:
    rng = random.Random(seed)
    values: defaultdict[State, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for epoch in range(epochs):
        held = 0
        exploration = max(0.05, 0.3 * (1 - epoch / epochs))
        for index, (observed, realized) in enumerate(zip(returns, returns[1:])):
            state = market_state(observed, held)
            action = rng.randrange(2) if rng.random() < exploration else int(values[state][1] > values[state][0])
            reward = transition(held, action, realized, cost)
            if index == len(returns) - 2:
                continuation = -action * cost
            else:
                continuation = 0.95 * max(values[market_state(realized, action)])
            values[state][action] += 0.08 * (reward + continuation - values[state][action])
            held = action

    def choose(state: State) -> int:
        scores = values.get(state, (0.0, 0.0))
        return int(scores[1] > scores[0])

    return choose


def replay(returns: list[float], policy: Policy, cost: float) -> ReplayResult:
    held, traded = 0, 0
    equity, peak, drawdown = 1.0, 1.0, 0.0
    for observed, realized in zip(returns, returns[1:]):
        target = policy(market_state(observed, held))
        equity *= 1 + transition(held, target, realized, cost)
        traded += abs(target - held)
        held = target
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)

    equity *= 1 - held * cost
    return ReplayResult(equity - 1, max(drawdown, 1 - equity / peak), traded + held)


if __name__ == "__main__":
    returns = synthetic_returns()
    train, test = returns[:240], returns[240:]
    cost = (FEE_BPS + SLIPPAGE_BPS) / 10_000
    results = {
        "q_learning": asdict(replay(test, train_policy(train, cost), cost)),
        "buy_and_hold": asdict(replay(test, lambda state: 1, cost)),
    }
    print(json.dumps({"dataset": "synthetic", "cost_bps": FEE_BPS + SLIPPAGE_BPS, "test_days": len(test), "results": results}, indent=2))
