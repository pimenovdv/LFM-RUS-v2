import pytest
from src.data_prep.filters import SpamLogCyclicFilter
from datatrove.data import Document

def test_seo_spam_filter():
    filter_obj = SpamLogCyclicFilter(remove_seo=True, remove_logs=False, remove_cyclic=False)

    # Should not filter normal text
    normal_doc = Document(text="This is a normal sentence about machine learning.", id="1")
    assert filter_obj.filter(normal_doc) == True

    # Should filter spam text with multiple keywords
    spam_doc = Document(text="Buy cheap things here! Click here to make money fast and get free download.", id="2")
    assert filter_obj.filter(spam_doc) == False

def test_log_filter():
    filter_obj = SpamLogCyclicFilter(remove_seo=False, remove_logs=True, remove_cyclic=False)

    normal_doc = Document(text="Line 1\nLine 2\nLine 3", id="1")
    assert filter_obj.filter(normal_doc) == True

    log_doc = Document(text="[Thread-1] INFO: App started\n2023-01-01 10:00:00 ERROR: Crash\n[Thread-2] DEBUG: Checking", id="2")
    assert filter_obj.filter(log_doc) == False

def test_cyclic_filter():
    filter_obj = SpamLogCyclicFilter(remove_seo=False, remove_logs=False, remove_cyclic=True)

    normal_doc = Document(text="Line 1\nLine 2\nLine 3\nLine 4\nLine 5", id="1")
    assert filter_obj.filter(normal_doc) == True

    cyclic_doc = Document(text="A\nA\nA\nA\nA\nA\nA\nA\nA\nA\nA\nA", id="2")
    assert filter_obj.filter(cyclic_doc) == False

    repeated_str = "print('hello world') " * 50
    repeated_doc = Document(text=repeated_str, id="3")
    assert filter_obj.filter(repeated_doc) == False
