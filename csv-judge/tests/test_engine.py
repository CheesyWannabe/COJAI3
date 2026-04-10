"""
Test Suite for CSV Comparison Engine
Tests: perfect match, no match, partial, shuffled rows/cols, edge cases.
"""

import pytest
import pandas as pd
import io
from engine import compare_csvs, cell_similarity, match_columns, normalize_value


def make_csv(data: list, columns: list) -> bytes:
    df = pd.DataFrame(data, columns=columns)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ─── cell_similarity ──────────────────────────────────────────────────────────

class TestCellSimilarity:
    def test_exact_match(self):
        assert cell_similarity("hello", "hello") == 1.0

    def test_empty_both(self):
        assert cell_similarity("", "") == 1.0

    def test_one_empty(self):
        assert cell_similarity("hello", "") == 0.0
        assert cell_similarity("", "world") == 0.0

    def test_numeric_exact(self):
        assert cell_similarity("3.14", "3.14") == 1.0

    def test_numeric_close(self):
        score = cell_similarity("3.14159", "3.14160")
        assert score > 0.9

    def test_numeric_far(self):
        score = cell_similarity("100", "1")
        assert score < 0.5

    def test_string_partial(self):
        score = cell_similarity("hello world", "hello")
        assert 0.3 < score < 1.0


# ─── Perfect Match ────────────────────────────────────────────────────────────

class TestPerfectMatch:
    def test_identical_csv(self):
        cols = ["name", "age", "city"]
        data = [["Alice", "30", "NYC"], ["Bob", "25", "LA"]]
        csv = make_csv(data, cols)
        result = compare_csvs(csv, csv)
        assert result["score"] == 100.0

    def test_single_row(self):
        cols = ["a", "b"]
        data = [["x", "y"]]
        csv = make_csv(data, cols)
        result = compare_csvs(csv, csv)
        assert result["score"] == 100.0

    def test_numeric_values(self):
        cols = ["val"]
        data = [["1.0"], ["2.5"], ["99.9"]]
        csv = make_csv(data, cols)
        result = compare_csvs(csv, csv)
        assert result["score"] == 100.0


# ─── No Match ─────────────────────────────────────────────────────────────────

class TestNoMatch:
    def test_completely_different(self):
        cols = ["a", "b", "c"]
        ref_data = [["x", "y", "z"], ["1", "2", "3"]]
        sub_data = [["p", "q", "r"], ["7", "8", "9"]]
        ref = make_csv(ref_data, cols)
        sub = make_csv(sub_data, cols)
        result = compare_csvs(ref, sub)
        assert result["score"] < 20.0

    def test_empty_submission(self):
        cols = ["a", "b"]
        ref = make_csv([["x", "y"]], cols)
        sub = make_csv([], cols)
        result = compare_csvs(ref, sub)
        assert result["score"] == 0.0


# ─── Shuffled Rows ────────────────────────────────────────────────────────────

class TestShuffledRows:
    def test_row_order_shuffled(self):
        cols = ["id", "name", "score"]
        ref_data = [["1", "Alice", "90"], ["2", "Bob", "85"], ["3", "Carol", "92"]]
        sub_data = [["3", "Carol", "92"], ["1", "Alice", "90"], ["2", "Bob", "85"]]
        ref = make_csv(ref_data, cols)
        sub = make_csv(sub_data, cols)
        result = compare_csvs(ref, sub)
        assert result["score"] >= 95.0, f"Expected >=95, got {result['score']}"

    def test_large_shuffle(self):
        import random
        cols = ["id", "val"]
        data = [[str(i), str(i * 2)] for i in range(50)]
        shuffled = data[:]
        random.shuffle(shuffled)
        ref = make_csv(data, cols)
        sub = make_csv(shuffled, cols)
        result = compare_csvs(ref, sub)
        assert result["score"] >= 95.0


# ─── Shuffled Columns ────────────────────────────────────────────────────────

class TestShuffledColumns:
    def test_column_order_shuffled(self):
        ref_cols = ["name", "age", "city"]
        sub_cols = ["city", "name", "age"]
        data = [["Alice", "30", "NYC"], ["Bob", "25", "LA"]]
        ref = make_csv(data, ref_cols)

        sub_data = [["NYC", "Alice", "30"], ["LA", "Bob", "25"]]
        sub = make_csv(sub_data, sub_cols)
        result = compare_csvs(ref, sub)
        assert result["score"] >= 95.0, f"Expected >=95, got {result['score']}"


# ─── Missing / Extra Rows ────────────────────────────────────────────────────

class TestMissingExtraRows:
    def test_missing_rows_in_submission(self):
        cols = ["a", "b"]
        ref_data = [["x", "1"], ["y", "2"], ["z", "3"]]
        sub_data = [["x", "1"], ["y", "2"]]  # missing last row
        ref = make_csv(ref_data, cols)
        sub = make_csv(sub_data, cols)
        result = compare_csvs(ref, sub)
        assert 0 < result["score"] < 100
        assert result["missing_rows"] == 1

    def test_extra_rows_in_submission(self):
        cols = ["a", "b"]
        ref_data = [["x", "1"]]
        sub_data = [["x", "1"], ["y", "2"], ["z", "3"]]
        ref = make_csv(ref_data, cols)
        sub = make_csv(sub_data, cols)
        result = compare_csvs(ref, sub)
        assert result["extra_rows"] == 2


# ─── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_whitespace_normalization(self):
        cols = ["name"]
        ref = make_csv([["  Alice  "]], cols)
        sub = make_csv([["Alice"]], cols)
        result = compare_csvs(ref, sub)
        assert result["score"] == 100.0

    def test_case_insensitive(self):
        cols = ["name"]
        ref = make_csv([["ALICE"]], cols)
        sub = make_csv([["alice"]], cols)
        result = compare_csvs(ref, sub, lowercase=True)
        assert result["score"] == 100.0

    def test_case_sensitive(self):
        cols = ["name"]
        ref = make_csv([["ALICE"]], cols)
        sub = make_csv([["alice"]], cols)
        result = compare_csvs(ref, sub, lowercase=False)
        assert result["score"] < 100.0

    def test_duplicate_rows(self):
        cols = ["a", "b"]
        ref_data = [["x", "1"], ["x", "1"]]
        sub_data = [["x", "1"], ["x", "1"]]
        ref = make_csv(ref_data, cols)
        sub = make_csv(sub_data, cols)
        result = compare_csvs(ref, sub)
        assert result["score"] == 100.0

    def test_single_column(self):
        cols = ["value"]
        ref = make_csv([["a"], ["b"], ["c"]], cols)
        sub = make_csv([["a"], ["b"], ["c"]], cols)
        result = compare_csvs(ref, sub)
        assert result["score"] == 100.0

    def test_numeric_tolerance(self):
        cols = ["value"]
        ref = make_csv([["3.14159265"]], cols)
        sub = make_csv([["3.14159266"]], cols)
        result = compare_csvs(ref, sub, numeric_tolerance=1e-5)
        assert result["score"] == 100.0

    def test_partial_match_intermediate(self):
        cols = ["a", "b", "c"]
        ref_data = [["x", "y", "z"], ["1", "2", "3"]]
        sub_data = [["x", "y", "WRONG"], ["1", "2", "3"]]
        ref = make_csv(ref_data, cols)
        sub = make_csv(sub_data, cols)
        result = compare_csvs(ref, sub)
        assert 50 < result["score"] < 100


# ─── Column Matching ──────────────────────────────────────────────────────────

class TestColumnMatching:
    def test_exact_header_match(self):
        mapping = match_columns(["name", "age"], ["age", "name"])
        assert mapping[0] == 1  # "name" → index 1 in sub
        assert mapping[1] == 0  # "age" → index 0 in sub

    def test_missing_column(self):
        mapping = match_columns(["name", "age", "city"], ["name", "age"])
        assert 0 in mapping
        assert 1 in mapping
        assert 2 not in mapping


# ─── Performance ─────────────────────────────────────────────────────────────

class TestPerformance:
    def test_large_csv_within_time(self):
        import time
        cols = [f"col{i}" for i in range(10)]
        data = [[str(i * j) for j in range(10)] for i in range(300)]
        csv = make_csv(data, cols)
        t0 = time.time()
        result = compare_csvs(csv, csv)
        elapsed = time.time() - t0
        assert result["score"] == 100.0
        assert elapsed < 30.0, f"Took too long: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
