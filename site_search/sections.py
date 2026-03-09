import logging
import re

from pydantic import BaseModel

logger = logging.getLogger("site_search")

# https://spec.commonmark.org/0.31.2/#atx-heading
section_pattern = re.compile(r"\s{0,3}(#+)\s+(.*?)\s*#*\s*")


class Section(BaseModel):
    title: str
    content: str
    url: str
    parents: list[str]
    level: int
    line: int


class _ParsingResult(BaseModel):
    sections: list[Section]
    links: list[str]


def _parse_markdown(document: str, url: str) -> _ParsingResult:
    lines = document.splitlines()

    # needs to be a dict because the first section does not have to be level 1
    last: dict[int, Section | None] = {}

    sections: list[Section] = []
    links: list[str] = []

    for lnum, line in enumerate(lines):
        # TODO: find links
        if match := section_pattern.fullmatch(line):
            level = len(match.group(1))
            title = match.group(2)

            # higher level sections end here
            for i in range(level + 1, 6):
                last[i] = None

            # lower level sections are parents
            parents: list[str] = []
            for i in range(1, level):
                if (parent := last.get(i)) is not None:
                    parents.append(parent.title)

            section = Section(
                title=title,
                content=line,
                url=url,
                parents=parents,
                level=level,
                line=lnum,
            )

            last[level] = section
            sections.append(section)
        else:
            # text before any heading
            if len(sections) == 0:
                sections.append(
                    Section(
                        title="",
                        content="",
                        url=url,
                        parents=[],
                        level=1,
                        line=lnum,
                    )
                )

            # append text to last section content
            sections[-1].content += "\n" + line

    return _ParsingResult(sections=sections, links=links)


def main():
    pass


if __name__ == "__main__":
    main()
