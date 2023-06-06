import re
import uuid


def limit_text(text: str, lim: int = 80):
    """
    >>> limit_text("hello world", 5)
    'hello...'

    >>> limit_text("hello world", 100)
    'hello world'

    :param text: Text to limit
    :param lim: Max length
    :return: Limited text
    """
    if len(text) > lim:
        return text[:lim] + "..."
    return text


def highlight_search_match(text: str, query: str, before="<b>", after="</b>"):
    """
    >>> highlight_search_match("hello world", "world")
    'hello <b>world</b>'

    >>> highlight_search_match("hello world", "hello")
    '<b>hello</b> world'

    >>> highlight_search_match("hello world", "hell")
    '<b>hell</b>o world'

    >>> highlight_search_match("Hello world", "hell")
    '<b>Hell</b>o world'

    >>> highlight_search_match("hello world", "foo")
    'hello world'

    >>> highlight_search_match("hello world", "ello")
    'hello world'


    :param text: Found string
    :param query: Search query
    :param before: Tag before match
    :param after: Tag after match
    :return: Highlighted string
    """
    # Replace matches only on word boundaries
    return re.compile(r"\b(" + re.escape(query) + ")", re.IGNORECASE).sub(before + '\\1' + after, text)


def section_hash(url, titles) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url + ''.join(titles)))
