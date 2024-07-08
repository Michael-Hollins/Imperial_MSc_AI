import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from load_raw_data import is_string_or_single_item_list
from load_raw_data import load_one_firm

class TestStringOrSingleItem:
    def test_is_string(self):
        assert is_string_or_single_item_list("hello") == True, "Failed on string input"

    def test_is_single_item_list(self):
        assert is_string_or_single_item_list(["hello"]) == True, "Failed on single item list input"

    def test_is_empty_list(self):
        assert is_string_or_single_item_list([]) == False, "Failed on empty list input"

    def test_is_multi_item_list(self):
        assert is_string_or_single_item_list(["hello", "world"]) == False, "Failed on multi item list input"

    def test_is_integer(self):
        assert is_string_or_single_item_list(123) == False, "Failed on integer input"

    def test_is_none(self):
        assert is_string_or_single_item_list(None) == False, "Failed on None input"

    def test_is_dict(self):
        assert is_string_or_single_item_list({"key": "value"}) == False, "Failed on dict input"

    def test_is_tuple(self):
        assert is_string_or_single_item_list(("hello",)) == False, "Failed on tuple input"

    def test_is_list_with_integer(self):
        assert is_string_or_single_item_list([1]) == False, "Failed on single integer item list input"

    def test_is_list_with_mixed_types(self):
        assert is_string_or_single_item_list(["hello", 1]) == False, "Failed on mixed type list input"

    def test_is_set(self):
        assert is_string_or_single_item_list({"hello"}) == False, "Failed on set input"
        
        
# TODO: Add other tests for other functions
        
if __name__ == "__main__":
    pytest.main()