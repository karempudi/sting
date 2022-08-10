from pathlib import Path

import pytest
import sting.utils.param_io as param_io
from sting.utils import types


def test_load_reference_param():
    """
    Check if you can load the referecne file correctly
    """
    param = param_io.load_reference()
    assert isinstance(param, dict)

@pytest.mark.parametrize("mode_missing", ['exclude', 'include'])
def test_autofill_dict(mode_missing):
    # Setup
    a = {
        'a': 1,
        'z': {'x': 4},
        'only_in_a': 2,
    }

    ref = {
        'a': 2,
        'b': None,
        'c': 3,
        'z': {'x': 5, 'y': 6},
    }
    # Run
    a_ = param_io.autofill_dict(a, ref, mode_missing=mode_missing)

    assert a_['a'] == 1
    assert a_['b'] is None
    assert a_['c'] == 3
    assert a_['z']['x'] == 4
    assert a_['z']['y'] == 6
    if mode_missing == 'exclude':
        # basically a_ shouldn't have anything that is not in reference
        assert 'only_in_a' not in a_.keys()
    elif mode_missing == 'include':
        assert a_['only_in_a'] == 2

# TODO
# wirte more tests for saving and loading params correctly later