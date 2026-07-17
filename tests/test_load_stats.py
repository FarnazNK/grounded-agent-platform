"""Tests for the benchmark's latency statistics.

The whole benchmark rests on these numbers. A p99 that is subtly wrong is
worse than no p99: it is quotable, authoritative-looking, and false.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "load"))

from stats import realtime_factor, summarize


def test_summarize_empty_is_safe():
    """An empty series must not divide by zero — scenarios can reject 100%."""
    s = summarize([])
    assert s.count == 0
    assert s.p99_ms == 0.0
    assert s.p99_reliable is False


def test_percentiles_are_real_observations_not_interpolations():
    """Nearest-rank, not linear interpolation.

    With 1..100 ms, p95 must be a value that actually occurred (95), not an
    interpolated 95.05 that no request ever experienced.
    """
    latencies = [i / 1000.0 for i in range(1, 101)]  # 1ms..100ms
    s = summarize(latencies)

    assert s.p50_ms == pytest.approx(50.0)
    assert s.p95_ms == pytest.approx(95.0)
    assert s.p99_ms == pytest.approx(99.0)
    assert s.min_ms == pytest.approx(1.0)
    assert s.max_ms == pytest.approx(100.0)


def test_percentile_ordering_holds():
    latencies = [0.001, 0.002, 0.005, 0.010, 0.050, 0.100, 0.500]
    s = summarize(latencies)
    assert s.min_ms <= s.p50_ms <= s.p90_ms <= s.p95_ms <= s.p99_ms <= s.max_ms


def test_p99_flagged_unreliable_below_100_samples():
    """p99 over 50 samples is the max, not a percentile. Say so."""
    assert summarize([0.01] * 50).p99_reliable is False
    assert summarize([0.01] * 100).p99_reliable is True


def test_tail_is_not_hidden_by_mean():
    """The reason we report percentiles at all: many fast requests and a few
    catastrophic ones produce a healthy-looking mean.

    Note the sample construction. With 99 fast + 1 slow, nearest-rank p99 is
    correctly the 99th ordered value — i.e. still fast — because only 1% of
    requests are slow and p99 asks "how bad is the 99th percentile", not
    "what is the worst". That is not a bug; it is the definition, and it is
    why `max` is reported alongside. To see the tail *in* p99 you need >1% of
    samples to be slow, which is what this test builds.
    """
    latencies = [0.01] * 95 + [5.0] * 5  # 5% slow
    s = summarize(latencies)

    assert s.mean_ms < 300  # mean still looks survivable
    assert s.p50_ms == pytest.approx(10.0)  # median says everything is fine
    assert s.p99_ms == pytest.approx(5000.0)  # the tail is visible here
    assert s.max_ms == pytest.approx(5000.0)


def test_p99_reflects_definition_not_worst_case():
    """A single outlier in 100 samples belongs to max, not p99.

    Locks in the semantics so nobody 'fixes' the percentile to report the
    worst observation and quietly makes every p99 in the report wrong.
    """
    latencies = [0.01] * 99 + [5.0]
    s = summarize(latencies)

    assert s.p99_ms == pytest.approx(10.0), "p99 must not be the max"
    assert s.max_ms == pytest.approx(5000.0), "max must show the outlier"


def test_realtime_factor():
    # 0.5s of compute for 2s of audio => RTF 0.25, four times faster than real time
    assert realtime_factor(0.5, 2.0) == pytest.approx(0.25)
    # slower than real time
    assert realtime_factor(4.0, 2.0) == pytest.approx(2.0)
    # guard against divide-by-zero on empty audio
    assert realtime_factor(1.0, 0.0) == 0.0
