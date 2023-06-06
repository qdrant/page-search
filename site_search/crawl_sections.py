import json
import multiprocessing
import os
import uuid
from dataclasses import dataclass
from typing import Optional, List, Iterable, Tuple
from urllib.parse import urlparse, urljoin

import requests
import tqdm
from bs4 import BeautifulSoup

from site_search.common import section_hash
from site_search.config import DATA_DIR
from site_search.crawl import Crawler, get_path_hierarchy, HEADER_TAGS, selector_soup


@dataclass
class SectionLine:
    text: str
    section_id: str
    location: str


@dataclass
class Section:
    url: str
    sections: Optional[List[str]] = None
    titles: Optional[List[str]] = None

    def get_id(self):
        # Generate hash for (url + titles) and convert it into a UUID
        return section_hash(self.url, self.titles)


class SectionCrawler(Crawler):

    def crawl_page_sections(self, url: str, content_selector="article") -> Iterable[Tuple[Section, List[SectionLine]]]:
        sections = get_path_hierarchy(url)

        resp = requests.get(url)
        if not resp.ok:
            return None

        soup = BeautifulSoup(resp.content, 'html.parser')

        title = soup.find('title')
        titles = []
        if title:
            titles.append(title.text)

        if content_selector:
            content = soup.select_one(content_selector)
        else:
            content = soup

        if content is None:
            return None

        if self.relative_urls:
            # Remove domain from url
            save_url = urlparse(url).path
        else:
            save_url = url

        abstracts = []
        current_section = None
        for tag in content.find_all(['p', 'li', *HEADER_TAGS]):
            tag_text = tag.text.strip()

            if tag.name in HEADER_TAGS:
                if current_section:
                    yield current_section, abstracts
                    abstracts = []

                current_section = Section(
                    url=save_url,
                    titles=titles + [tag.text],
                    sections=sections
                )
                continue
            else:
                if current_section is None:
                    continue

            if self.relative_urls:
                # Remove domain from url
                save_url = urlparse(url).path
            else:
                save_url = url

            if self.split_lines:
                lines = tag_text.splitlines()
            else:
                lines = [tag_text]

            for line in lines:
                if line:
                    abstracts.append(SectionLine(
                        text=line,
                        section_id=current_section.get_id(),
                        location=selector_soup(tag),
                    ))

        if current_section:
            yield current_section, abstracts

    def crawl_page_sections_list(self, url: str, content_selector="article") -> List[Tuple[Section, List[SectionLine]]]:
        return list(self.crawl_page_sections(url, content_selector))


def download_and_save(abstracts_name='section_abstracts.jsonl', section_name="sections.jsonl", split_lines=True):
    page_url = "https://qdrant.tech/"
    site_map_url = page_url + "sitemap.xml"
    crawler = SectionCrawler(page_url, split_lines=split_lines)

    pages = crawler.download_sitemap(site_map_url)

    with open(os.path.join(DATA_DIR, abstracts_name), 'w') as out_abstracts:
        with open(os.path.join(DATA_DIR, section_name), 'w') as out_sections:
            page_urls = []
            for page in pages:
                full_page_url = urljoin(page_url, page.url)
                page_urls.append(full_page_url)

            with multiprocessing.Pool(processes=10) as pool:
                for abstracts in tqdm.tqdm(pool.imap(crawler.crawl_page_sections_list, page_urls)):
                    for (section, abstracts) in abstracts:
                        out_sections.write(json.dumps(section.__dict__))
                        out_sections.write('\n')
                        for abstract in abstracts:
                            out_abstracts.write(json.dumps(abstract.__dict__))
                            out_abstracts.write('\n')


if __name__ == '__main__':
    download_and_save(
        abstracts_name='section_abstracts.jsonl',
        section_name="sections.jsonl",
        split_lines=True,
    )
