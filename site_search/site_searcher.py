from site_search.neural_searcher import NeuralSearcher


class SiteSearcher:
    def __init__(self, collection_name):
        self.neural_searcher = NeuralSearcher(collection_name=collection_name)

    def _get_section_filter(self, section):
        return {
            "must": [
                {"key": "sections", "match": {"value": section}},
            ]
        } if section else None

    def _neural_search(self, text, section):
        section_filter = self._get_section_filter(section=section)
        return {
            "result": self.neural_searcher.search(text=text, filter_=section_filter)
        }

    def _prefix_search(self, text, section):


    def search(self, text, section=None):
        section_filter = {
            "must": [
                {"key": "sections", "match": {"value": section}},
            ]
        } if section else None

        return {
            "result": self.neural_searcher.search(text=text, filter_=section_filter)
        }
