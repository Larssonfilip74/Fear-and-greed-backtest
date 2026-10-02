from datetime import date

import pandas as pd
import pytest

from fgbt.splits import HoldoutLockedError, assert_no_holdout, research_view, segment


def _df():
    return pd.DataFrame({"x": [1, 2, 3]}, index=[date(2020, 12, 31), date(2024, 12, 31), date(2025, 1, 2)])


def test_research_view_drops_holdout_by_default():
    assert list(research_view(_df()).index) == [date(2020, 12, 31), date(2024, 12, 31)]
    assert len(research_view(_df(), unlock_holdout=True)) == 3


def test_assert_no_holdout_raises():
    with pytest.raises(HoldoutLockedError):
        assert_no_holdout(_df().index)
    assert_no_holdout(_df().index, unlock_holdout=True)


@pytest.mark.parametrize("d,seg", [(date(2011, 1, 3), "DEV"), (date(2021, 1, 15), "NONE"),
                                   (date(2021, 2, 1), "VAL"), (date(2025, 1, 2), "HOLD")])
def test_segment(d, seg):
    assert segment(d) == seg
