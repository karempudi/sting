from copy import deepcopy
from sting.utils.types import RecursiveNamespace
import pytest
# This test is copied from 
# https://github.com/TuragaLab/DECODE/blob/master/decode/test/test_types.py

class TestRecursiveNamespace:

    d = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3,
        }
    }

    dr = RecursiveNamespace()
    dr.a = 1
    dr.b = RecursiveNamespace()
    dr.b.c = 2
    dr.b.d = 3

    def test_dict2namespace(self):

        """Run"""
        dr = RecursiveNamespace(**self.d)

        """Assertions"""
        assert dr.a == 1
        assert dr.b.c == 2
        assert dr.b.d == 3

    def test_namespace2dict(self):
        """Run"""
        d = self.dr.to_dict()

        """Assertions"""
        assert d['a'] == 1
        assert d['b']['c'] == 2
        assert d['b']['d'] == 3

    def test_cycle(self):

        """Run and Assert"""
        assert RecursiveNamespace(**self.d).to_dict() == self.d
        assert RecursiveNamespace(**self.dr.to_dict()) == self.dr

    def test_mapping(self):
        
        with pytest.raises(TypeError):
            x = dict(**self.dr) # can't do this so will, raise exception

        x = dict(**self.dr.b)
        assert x['c'] == 2
        assert x['d'] == 3