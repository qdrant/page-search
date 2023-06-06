from site_search.crawl import download_and_save

if __name__ == '__main__':
    download_and_save(
        file_name='abstracts_large.jsonl',
        split_lines=False,
    )

