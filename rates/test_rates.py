"""Focused tests of port-centered selection and physical-port retention."""

import random
import unittest
from collections import defaultdict

import igraph as ig

from run_rates import add_port_bypasses, area, port_curve
from robustness_experiment import HistoricalStates, neighbors


class PortRatesTests(unittest.TestCase):
    def test_only_focal_port_critical_state_generates_bypass(self):
        labels = ["1|", "2|", "3|", "3|1"]
        states = HistoricalStates(labels)
        existing = {(0, 1), (1, 2)}
        incoming, outgoing = neighbors(len(labels), existing)
        rank = defaultdict(list, {"2": [(1, 1)]})
        args = (rank, states, existing, incoming, outgoing,
                [len(x) for x in incoming], [len(x) for x in outgoing])
        added, trace = add_port_bypasses("2", "CB-LDF", 1, random.Random(42), *args)
        self.assertEqual(added, [(0, 3)])
        self.assertEqual(trace[0]["critical_state"], "2|")
        self.assertEqual(add_port_bypasses("1", "CB-LDF", 1,
                                           random.Random(42), *args)[0], [])

    def test_single_port_component_does_not_count_as_global_retention(self):
        topology = ig.Graph(n=2, edges=[(0, 1), (1, 0)], directed=True)
        values = port_curve(topology, [[0, 1], [0]], [0], 1)
        self.assertEqual(values, [1.0, 0.0])
        self.assertAlmostEqual(area(values, [0, 1], 2), 0.25)


if __name__ == "__main__":
    unittest.main()
