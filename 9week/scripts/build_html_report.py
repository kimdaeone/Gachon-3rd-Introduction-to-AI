"""Build the seven-page offline WISDM research report from its HTML fragments."""

import argparse
from html import escape, unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from shutil import copyfile
from urllib.parse import quote, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "scripts" / "html_parts"
PAGES = (
    ("index", "연구 개요", "스마트폰 가속도계 기반 사람 활동 인식"),
    ("plan", "연구 계획", "연구 설계와 실행"),
    ("data", "데이터", "데이터와 평가"),
    ("models", "모델 설계", "모델 구조와 학습"),
    ("development", "개발 검증", "1단계 개발 검증 결과"),
    ("test", "최종 테스트", "최종 테스트"),
    ("review", "결론과 한계", "결론과 한계"),
)
LINK_ATTR = re.compile(r'''(\b(?:href|src)\s*=\s*)(["'])(.*?)\2''', re.IGNORECASE)
MARKDOWN_LINK = re.compile(r"\]\((<[^>]+>|[^\s)]+)")
TABLE_CAPTION = re.compile(r'(<div class="table-wrap")(>\s*<table>\s*<caption>)([^<]+)')


class Outline(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sections = []
        self.current = None
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        anchor = attributes.get("id")
        if anchor:
            if anchor in self.ids:
                raise ValueError(f"Duplicate HTML id: {anchor}")
            self.ids.add(anchor)
        if tag == "h2":
            self.current = (anchor, [])
        self.links.extend(value for key, value in attrs if key in {"href", "src"})

    def handle_data(self, data):
        if self.current:
            self.current[1].append(data)

    def handle_endtag(self, tag):
        if tag == "h2" and self.current:
            anchor, words = self.current
            if anchor:
                self.sections.append((anchor, "".join(words).strip()))
            self.current = None


def collect_source(source, sources):
    """Keep project-relative paths so links among the original Markdown files work."""
    relative = source.relative_to(ROOT)
    if relative in sources:
        return relative
    if not source.is_file():
        raise FileNotFoundError(source)
    sources.add(relative)
    if source.suffix.lower() == ".md":
        for link in MARKDOWN_LINK.findall(source.read_text(encoding="utf-8-sig")):
            url = urlsplit(link.strip("<>"))
            if not url.scheme and not url.netloc and url.path:
                target = (source.parent / unquote(url.path)).resolve()
                collect_source(target, sources)
    return relative


def rewrite_sources(fragment, sources):
    def replace(match):
        prefix, delimiter, value = match.groups()
        url = urlsplit(value)
        if url.scheme or url.netloc or not url.path.startswith("../"):
            return match.group()
        relative = collect_source((ROOT / unquote(url.path[3:])).resolve(), sources)
        destination = f"sources/{quote(relative.as_posix())}"
        url = url._replace(path=destination).geturl()
        return f"{prefix}{delimiter}{url}{delimiter}"

    return LINK_ATTR.sub(replace, fragment)


def page_navigation(index):
    links = []
    for offset, label, arrow in ((-1, "이전 장", "←"), (1, "다음 장", "→")):
        if 0 <= index + offset < len(PAGES):
            key, chapter, _ = PAGES[index + offset]
        else:
            continue
        links.append(
            f'<a href="{key}.html" class="page-turn" rel="{"prev" if offset < 0 else "next"}">'
            f'<span class="page-turn-label"><span aria-hidden="true">{arrow}</span> {label}</span>'
            f'<strong>{chapter}</strong></a>'
        )
    return '<nav class="page-turns" aria-label="이전 및 다음 장">' + "".join(links) + "</nav>"


def label_scrollable_tables(fragment):
    def add_region(match):
        container, opening, caption = match.groups()
        label = escape(unescape(caption).strip())
        return f'{container} tabindex="0" role="region" aria-label="{label}"{opening}{caption}'

    return TABLE_CAPTION.sub(add_region, fragment)


def make_page(index, fragment):
    key, chapter, title = PAGES[index]
    outline = Outline()
    outline.feed(fragment)
    navigation_items = []
    for slug, label, _ in PAGES:
        current = ' aria-current="page"' if slug == key else ""
        navigation_items.append(f'<li><a href="{slug}.html"{current}>{label}</a></li>')
    navigation = "\n".join(navigation_items)
    contents = ""
    if outline.sections:
        section_links = "\n".join(
            f'<li><a href="#{quote(anchor)}">{escape(label)}</a></li>'
            for anchor, label in outline.sections
        )
        contents = f'''<nav class="section-index" aria-label="이 장의 목차">
      <p class="section-index-title">이 장의 목차</p>
      <ul>{section_links}</ul>
    </nav>'''

    hero_class = 'page-hero page-hero-home' if key == 'index' else 'page-hero'
    layout_class = 'reading-layout frame' if contents else 'reading-layout frame single-column'
    return f'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="가천대학교 기계학습프로그래밍 수업의 WISDM 1D CNN 사람 활동 인식 연구 보고서">
  <title>{title} | WISDM 연구 보고서</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <a class="skip-link" href="#report-content">본문 바로가기</a>
  <header class="masthead" id="top">
    <div class="masthead-inner frame">
      <a class="wordmark" href="index.html">WISDM 연구 보고서</a>
      <p class="course">가천대학교 · 기계학습프로그래밍</p>
    </div>
    <nav class="site-nav" aria-label="보고서 장">
      <ul class="frame">{navigation}</ul>
    </nav>
  </header>
  <main id="report-content" tabindex="-1">
    <div class="{hero_class}">
      <div class="frame">
        <p class="page-location">{chapter} <span aria-hidden="true">/</span> <span>{index + 1} / {len(PAGES)}장</span></p>
        <h1>{title}</h1>
      </div>
    </div>
    <div class="{layout_class}">
      {contents}
      <div class="report-body">
{fragment.rstrip()}
        {page_navigation(index)}
      </div>
    </div>
  </main>
  <footer class="report-footer">
    <div class="frame footer-inner">
      <p>가천대학교 · 기계학습프로그래밍 연구 보고서</p>
      <a href="#top">맨 위로 <span aria-hidden="true">↑</span></a>
    </div>
  </footer>
</body>
</html>
'''


def check_links(pages, source_files):
    available = {"style.css", *pages, *(f"sources/{path.as_posix()}" for path in source_files)}
    for page_name, document in pages.items():
        for link in document.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            path = unquote(url.path) or page_name
            if path not in available:
                raise ValueError(f"Broken link in {page_name}: {link}")
            if url.fragment and path in pages and unquote(url.fragment) not in pages[path].ids:
                raise ValueError(f"Missing anchor in {page_name}: {link}")


def build(output):
    sources = set()
    pages = {}
    for index, (key, _, _) in enumerate(PAGES):
        fragment = (PARTS / f"{key}.html").read_text(encoding="utf-8-sig")
        fragment = label_scrollable_tables(rewrite_sources(fragment, sources))
        pages[f"{key}.html"] = make_page(index, fragment)

    outlines = {}
    for file_name, page in pages.items():
        outlines[file_name] = Outline()
        outlines[file_name].feed(page)
    check_links(outlines, sources)

    output.mkdir(parents=True, exist_ok=True)
    copyfile(PARTS / "style.css", output / "style.css")
    for relative in sorted(sources):
        destination = output / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(ROOT / relative, destination)
    for file_name, page in pages.items():
        (output / file_name).write_text(page, encoding="utf-8", newline="\n")
    print(f"Built {len(pages)} pages with {len(sources)} offline source files in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "html_report")
    build(parser.parse_args().output.resolve())
