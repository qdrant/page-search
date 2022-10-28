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

    def _prefix_search(self, text, section, headers=True):
        tags = ["h1", "h2", "h3", "h4", "h5", "h6"] if headers else ["p", "li"]
        return self.text_searcher.search(text=text, section=section, tags=tags)

    def search(self, text, section=None):
        prefix_results = self._prefix_search(text=text, section=section)
        additional_results = []

        if len(prefix_results) < SEARCH_LIMIT:
            if len(text) > 3:
                additional_results = self._neural_search(text=text, section=section)
            else:
                additional_results = self._prefix_search(text=text, section=section, headers=False)

        return prefix_results + additional_results[:SEARCH_LIMIT - len(prefix_results)]
