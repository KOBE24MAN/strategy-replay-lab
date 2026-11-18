import unittest

from market_fixture import synthetic_returns
from strategy_replay import replay, train_policy


class StrategyReplayTests(unittest.TestCase):
    def test_baseline_pays_entry_and_exit_cost_without_using_first_return(self):
        result = replay([0.9, 0.10, -0.05], lambda state: 1, 0.01)
        self.assertAlmostEqual(result.net_return, (1 + 0.10 - 0.01) * 0.95 * 0.99 - 1)
        self.assertEqual(result.traded_units, 2)

    def test_seeded_policy_replay_is_repeatable_on_holdout(self):
        series = synthetic_returns()
        training, holdout = series[:240], series[240:]
        first = replay(holdout, train_policy(training, 0.0008), 0.0008)
        second = replay(holdout, train_policy(training, 0.0008), 0.0008)
        self.assertEqual(first, second)
