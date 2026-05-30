from src.data_prep.filters import SpamLogCyclicFilter
from datatrove.data import Document

def test_seo_spam_filter():
    filter_obj = SpamLogCyclicFilter(remove_seo=True, remove_logs=False, remove_cyclic=False)

    # Should not filter normal text
    normal_doc = Document(text="This is a normal sentence about machine learning.", id="1")
    assert filter_obj.filter(normal_doc)

    # Should filter spam text with multiple keywords
    spam_doc = Document(text="Buy cheap things here! Click here to make money fast and get free download.", id="2")
    assert not filter_obj.filter(spam_doc)

def test_log_filter():
    filter_obj = SpamLogCyclicFilter(remove_seo=False, remove_logs=True, remove_cyclic=False)

    normal_doc = Document(text="Line 1\nLine 2\nLine 3", id="1")
    assert filter_obj.filter(normal_doc)

    log_doc = Document(text="[Thread-1] INFO: App started\n2023-01-01 10:00:00 ERROR: Crash\n[Thread-2] DEBUG: Checking", id="2")
    assert not filter_obj.filter(log_doc)

def test_cyclic_filter():
    filter_obj = SpamLogCyclicFilter(remove_seo=False, remove_logs=False, remove_cyclic=True)

    normal_doc = Document(text="Line 1\nLine 2\nLine 3\nLine 4\nLine 5", id="1")
    assert filter_obj.filter(normal_doc)

    cyclic_doc = Document(text="A\nA\nA\nA\nA\nA\nA\nA\nA\nA\nA\nA", id="2")
    assert not filter_obj.filter(cyclic_doc)

    repeated_str = "print('hello world') " * 50
    repeated_doc = Document(text=repeated_str, id="3")
    assert not filter_obj.filter(repeated_doc)

from src.data_prep.filters import OpenAIClassifierFilter
import json

def test_openai_classifier_filter(mocker):
    good_doc = Document(text="This is an excellent high quality text.", id="1")
    bad_doc = Document(text="sdkfjhsdkjfhdsf bad quality.", id="2")
    error_doc = Document(text="trigger error", id="3")

    filter_obj = OpenAIClassifierFilter(
        model_name="test-model",
        prompt="Classify",
        good_label="good",
        bad_label="bad",
        api_key="fake-key"
    )

    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, message):
            self.message = message

    class MockResponse:
        def __init__(self, choices):
            self.choices = choices

    def mock_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        text = messages[1]["content"] if len(messages) > 1 else ""
        if "excellent" in text:
            return MockResponse([MockChoice(MockMessage(json.dumps({"label": "good"})))])
        elif "bad" in text:
            return MockResponse([MockChoice(MockMessage(json.dumps({"label": "bad"})))])
        else:
            raise Exception("API error")

    mocker.patch("openai.resources.chat.completions.Completions.create", side_effect=mock_create)

    results = filter_obj.filter_batch([good_doc, bad_doc, error_doc])

    assert results[0]
    assert good_doc.metadata["openai_label"] == "good"

    assert isinstance(results[1], tuple)
    assert not results[1][0]
    assert "openai classified as bad" in results[1][1]
    assert bad_doc.metadata["openai_label"] == "bad"

    assert isinstance(results[2], tuple)
    assert not results[2][0]
    assert "openai api error" in results[2][1]
