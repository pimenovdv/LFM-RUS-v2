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
