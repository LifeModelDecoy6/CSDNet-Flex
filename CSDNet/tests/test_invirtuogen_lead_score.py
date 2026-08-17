import os
import tempfile
import unittest

import pandas as pd

from CSDNet.exp.lead.invirtuogen_score import aggregate_cells, load_task_run


class InVirtuoGenLeadScoreTest(unittest.TestCase):
    def write_result(self, directory, seed, rows):
        path = os.path.join(directory, f"parp1_id0_thr0.4_{seed}.csv")
        pd.DataFrame(rows).to_csv(path, index=False, header=False)
        return path

    def test_non_improving_constrained_candidate_only_enters_parenthetical_sum(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_result(
                directory,
                0,
                [["CCO", 7.2, 0.7, 0.8, 0.5, ""]],
            )
            row = load_task_run(path, 1000, {("parp1", 0): 7.3})

        self.assertFalse(row["strict_success"])
        self.assertEqual(row["strict_ds_signed"], 0.0)
        self.assertEqual(row["constraint_only_ds_signed"], -7.2)
        self.assertAlmostEqual(row["dock_shortfall"], 0.1)

    def test_cell_mean_counts_failed_random_run_as_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            failed = self.write_result(
                directory,
                0,
                [["CCO", 7.2, 0.7, 0.8, 0.5, ""]],
            )
            successful = self.write_result(
                directory,
                1,
                [["CCN", 8.0, 0.7, 0.8, 0.5, ""]],
            )
            rows = [
                load_task_run(failed, 1000, {("parp1", 0): 7.3}),
                load_task_run(successful, 1000, {("parp1", 0): 7.3}),
            ]
            cells = aggregate_cells(pd.DataFrame(rows))

        self.assertEqual(len(cells), 1)
        self.assertEqual(cells.iloc[0]["strict_success_runs"], 1)
        self.assertAlmostEqual(cells.iloc[0]["strict_ds_signed_mean"], -4.0)
        self.assertAlmostEqual(
            cells.iloc[0]["constraint_only_ds_signed_mean"],
            -7.6,
        )


if __name__ == "__main__":
    unittest.main()
