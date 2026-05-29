import re
from datatrove.pipeline.filters.base_filter import BaseFilter

class SpamLogCyclicFilter(BaseFilter):
    def __init__(
        self,
        remove_seo=True,
        remove_logs=True,
        remove_cyclic=True,
        exclusion_writer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.remove_seo = remove_seo
        self.remove_logs = remove_logs
        self.remove_cyclic = remove_cyclic
        self.exclusion_writer = exclusion_writer

        self.seo_keywords = [
            "buy cheap", "click here", "subscribe now", "free download",
            "make money fast", "casino", "viagra", "online pharmacy",
            "SEO optimization", "best price", "discount code"
        ]

    def _is_seo_spam(self, text):
        text_lower = text.lower()
        keyword_count = sum(1 for kw in self.seo_keywords if kw in text_lower)
        # If too many distinct spam keywords, flag it
        if keyword_count >= 3:
            return True
        return False

    def _is_log(self, text):
        lines = text.split('\n')
        if not lines:
            return False

        log_patterns = [
            r'INFO|DEBUG|WARN(?:ING)?|ERROR|FATAL|TRACE',
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # Datetime
            r'\[.*?\]'  # Common log prefixes like [Thread-1]
        ]

        log_line_count = 0
        for line in lines:
            # Check if line looks like a log
            if any(re.search(p, line, re.IGNORECASE) for p in log_patterns):
                log_line_count += 1

        # If more than 50% of lines look like logs (and at least 3 lines)
        if len(lines) >= 3 and log_line_count / len(lines) > 0.5:
            return True
        return False

    def _is_cyclic(self, text):
        lines = text.split('\n')
        if len(lines) < 4 and len(text) < 100:
            return False

        unique_lines = set(line.strip() for line in lines if line.strip())
        total_non_empty = sum(1 for line in lines if line.strip())

        if total_non_empty == 0:
            return False

        # If very few unique lines, it's highly repetitive
        if len(unique_lines) / total_non_empty < 0.2 and total_non_empty > 10:
            return True

        # N-gram repetition check (simple character-level for fast cyclic detection)
        n = 10
        if len(text) > n * 3:
            for i in range(len(text) // 2):
                chunk = text[i:i+n]
                if text.count(chunk) > 5:  # Same chunk repeats many times
                    return True
        return False

    def filter(self, doc):
        text = doc.text

        if self.remove_seo and self._is_seo_spam(text):
            return False

        if self.remove_logs and self._is_log(text):
            return False

        if self.remove_cyclic and self._is_cyclic(text):
            return False

        return True

from transformers import pipeline
import torch
from typing import Tuple, List, Union

class TransformersClassifierFilter(BaseFilter):
    def __init__(
        self,
        model_name: str,
        keep_labels: Union[Tuple[str, float], List[Tuple[str, float]], None] = None,
        remove_labels: Union[Tuple[str, float], List[Tuple[str, float]], None] = None,
        batch_size: int = 16,
        device: str = None,
        exclusion_writer = None,
        **kwargs
    ):
        super().__init__(exclusion_writer=exclusion_writer, batch_size=batch_size, **kwargs)
        self.model_name = model_name

        if keep_labels and remove_labels:
            raise ValueError("You can only specify one of keep_labels or remove_labels")

        if keep_labels and isinstance(keep_labels, tuple):
            keep_labels = [keep_labels]
        if remove_labels and isinstance(remove_labels, tuple):
            remove_labels = [remove_labels]

        self.keep_labels = keep_labels
        self.remove_labels = remove_labels

        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        self._pipeline = None

    @property
    def classifier(self):
        if self._pipeline is None:
            # Initialize pipeline lazily to avoid loading it on main process if using multiprocessing
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                truncation=True,
                max_length=512
            )
        return self._pipeline

    def filter(self, doc):
        # BaseFilter requires this to be implemented, but we rely on filter_batch
        raise NotImplementedError("Use filter_batch instead")

    def filter_batch(self, batch):
        texts = [doc.text for doc in batch]
        # Transformers pipeline natively handles batching if passed a list
        # Note: Depending on transformers version, might need to specify batch_size=len(texts) to pipeline call itself
        predictions = self.classifier(texts, batch_size=len(texts))

        results = []
        for doc, pred in zip(batch, predictions):
            label = pred['label']
            score = pred['score']

            doc.metadata["classifier_label"] = label
            doc.metadata["classifier_score"] = score

            keep = True
            reason = None

            if self.keep_labels:
                keep = False
                for k_label, k_score in self.keep_labels:
                    if label == k_label and score >= k_score:
                        keep = True
                        break
                if not keep:
                    reason = f"label {label} ({score:.2f}) not in keep_labels thresholds"

            elif self.remove_labels:
                for r_label, r_score in self.remove_labels:
                    if label == r_label and score >= r_score:
                        keep = False
                        reason = f"label {label} ({score:.2f}) meets remove_label threshold"
                        break

            if keep:
                results.append(True)
            else:
                results.append((False, reason))

        return results
