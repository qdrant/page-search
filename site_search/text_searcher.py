import re

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

from site_search.config import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, COLLECTION_NAME


class TextSearcher:
    def __init__(self, collection_name=COLLECTION_NAME):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True
        )

    @classmethod
    def highlight_search_match(cls, text: str, query: str, before="<b>", after="</b>"):
        """
        >>> TextSearcher.highlight_search_match("hello world", "world")
        'hello <b>world</b>'

        >>> TextSearcher.highlight_search_match("hello world", "hello")
        '<b>hello</b> world'

        >>> TextSearcher.highlight_search_match("hello world", "hell")
        '<b>hell</b>o world'

        >>> TextSearcher.highlight_search_match("hello world", "foo")
        'hello world'

        >>> TextSearcher.highlight_search_match("hello world", "ello")
        'hello world'


        :param text: Found string
        :param query: Search query
        :param before: Tag before match
        :param after: Tag after match
        :return: Highlighted string
        """
        # Replace matches only on word boundaries
        return re.compile(r"\b" + re.escape(query)).sub(before + query + after, text)

    def search(self, text, tags=None, section=None, filter_=None):
        scroll_filter = {
            "must": [
            ]
        }

        text_filter = {
            "key": "text",
            "match": {"text": text}
        }

        scroll_filter["must"].append(text_filter)

        if section:
            scroll_filter["must"].append({
                "key": "sections",
                "match": {"value": section}
            })

        if tags:
            tags_filter = {
                "should": [
                    {
                        "key": "tag",
                        "match": {"value": tag}
                    } for tag in tags
                ]
            }
            scroll_filter["must"].append(tags_filter)

        if filter_:
            scroll_filter["must"].append(filter_)

        search_result, _next_page = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(**scroll_filter),
            limit=5,
            with_payload=True,
            with_vectors=False,
        )
        payloads = [{
            "payload": hit.payload,
            "highlight": self.highlight_search_match(hit.payload['text'], text),
        } for hit in search_result]
        return payloads


if __name__ == '__main__':
    searcher = TextSearcher()
    for hit in searcher.search("reco", tags=["h1", "h2", "h3", "h4", "h5", "h6"], section="documentation"):
        print(hit)
