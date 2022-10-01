from site_search.config import SEARCH_LIMIT
from site_search.neural_searcher import NeuralSearcher
from site_search.text_searcher import TextSearcher


class SiteSearcher:
    def __init__(self, collection_name):
        self.neural_searcher = NeuralSearcher(collection_name=collection_name)
        self.text_searcher = TextSearcher(collection_name=collection_name)

    def _get_section_filter(self, section):
        return {
            "must": [
                {"key": "sections", "match": {"value": section}},
            ]
        } if section else None

    def _neural_search(self, text, section):
        section_filter = self._get_section_filter(section=section)
        return self.neural_searcher.search(text=text, filter_=section_filter)

    def _prefix_search(self, text, section):
        return self.text_searcher.search(text=text, section=section, tags=["h1", "h2", "h3", "h4", "h5", "h6"])

    def search(self, text, section=None):
        if len(text) > 3:
            prefix_results = self._prefix_search(text=text, section=section)
            if len(prefix_results) < SEARCH_LIMIT:
                return prefix_results + self._neural_search(text=text, section=section)[:SEARCH_LIMIT - len(prefix_results)]
            return prefix_results
        else:
            return self._prefix_search(text=text, section=section)
