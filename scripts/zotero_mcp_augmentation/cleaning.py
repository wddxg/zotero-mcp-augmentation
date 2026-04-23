from __future__ import annotations

import re

from .models import HeadingInfo


class MarkdownCleaner:
    CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
    TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")
    SPACE_RE = re.compile(r"\s+")
    TOC_LINE_RE = re.compile(
        r"^(?:"
        r"(?:第[0-9一二三四五六七八九十百零〇两]+[章节篇部]\s*.*)"
        r"|(?:[0-9]+(?:\.[0-9]+){0,4}\s+.+)"
        r"|(?:附录[A-Za-z0-9一二三四五六七八九十]*\s*.*)"
        r"|(?:参考文献|References|致谢|Acknowledg(?:e)?ments?)"
        r")(?:\s+|\.{2,}|…{2,}).*\d+\s*$",
        re.IGNORECASE,
    )
    METADATA_LINE_RE = re.compile(
        r"^(?:"
        r"DOI[:：].*|ISSN[:：].*|ISBN[:：].*|"
        r"中图分类号.*|文献标志码.*|文章编号.*|基金项目.*|收稿日期.*|"
        r"Article\s+info.*|ARTICLE\s+INFO.*|"
        r"作者简介.*|通信作者.*|网络首发.*|开放科学.*标识码.*|OSID.*|扫一扫二维码.*"
        r")$",
        re.IGNORECASE,
    )
    FRONT_MATTER_SIGNAL_RE = re.compile(
        r"(学位论文|A\s*Dissertation\s*Submitted|学校代码|研究生学号|原创性声明|独创性声明|Article\s+info|ARTICLE\s+INFO)",
        re.IGNORECASE,
    )
    STOP_HEADING_RE = re.compile(
        r"^(?:#+\s*)?(?:参考文献|References|致谢|Acknowledg(?:e)?ments?|Appendix(?:es)?|附录)\s*$",
        re.IGNORECASE,
    )
    SECTION_HEADING_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
        "abstract_zh": (
            re.compile(r"^(?:#+\s*)?摘\s*要\s*$"),
            re.compile(r"^(?:#+\s*)?中文摘要\s*$"),
        ),
        "abstract_en": (
            re.compile(r"^(?:#+\s*)?abstract\s*$", re.IGNORECASE),
            re.compile(r"^(?:#+\s*)?a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*$", re.IGNORECASE),
        ),
        "keywords_zh": (re.compile(r"^(?:#+\s*)?关键[词字]\s*$"),),
        "keywords_en": (
            re.compile(r"^(?:#+\s*)?key\s*words?\s*$", re.IGNORECASE),
            re.compile(r"^(?:#+\s*)?k\s*e\s*y\s*w\s*o\s*r\s*d\s*s\s*$", re.IGNORECASE),
        ),
        "introduction": (
            re.compile(r"^(?:#+\s*)?(?:\d+[.\-、]?)?\s*(?:引言|前言|绪论)\s*$"),
            re.compile(r"^(?:#+\s*)?(?:\d+[.\-、]?)?\s*introduction\s*$", re.IGNORECASE),
        ),
        "conclusion": (
            re.compile(r"^(?:#+\s*)?(?:\d+[.\-、]?)?\s*(?:结论|结语|结论与展望|研究结论|总结与展望)\s*$"),
            re.compile(r"^(?:#+\s*)?(?:\d+[.\-、]?)?\s*conclusions?\s*$", re.IGNORECASE),
        ),
    }
    BODY_HEADING_PATTERNS = (
        ("chapter", re.compile(r"^\s*(?P<label>第[一二三四五六七八九十百零〇两0-9]+章)\s*(?P<title>[^\n]{1,60})$")),
        ("hierarchical", re.compile(r"^\s*(?P<label>[1-9][0-9０-９]*(?:[.．][0-9０-９]+){1,3})\s*(?P<title>[^\n]{1,60})$")),
        ("arabic_simple", re.compile(r"^\s*(?P<label>[1-9][0-9０-９]*)(?:\s*[.．、\-]\s*)?(?P<title>[^\n]{2,60})$")),
    )
    BODY_REJECT_RE = re.compile(r"^(?:图|表|fig(?:ure)?|table)\s*\d+", re.IGNORECASE)
    FULLWIDTH_HEADING_TRANSLATION = str.maketrans("０１２３４５６７８９．", "0123456789.")

    @classmethod
    def normalize_line(cls, line: str) -> str:
        text = cls.CONTROL_CHAR_RE.sub("", line)
        text = text.replace("\u3000", " ").replace("\xa0", " ").replace("\u200b", "")
        text = re.sub(r"[ \t]+", " ", text)
        return text.rstrip()

    @staticmethod
    def collapse_cjk_spaces(text: str) -> str:
        previous = None
        while previous != text:
            previous = text
            text = re.sub(r"([\u3400-\u4dbf\u4e00-\u9fff]) +([\u3400-\u4dbf\u4e00-\u9fff])", r"\1\2", text)
        return text

    @staticmethod
    def strip_heading_markup(text: str) -> str:
        return re.sub(r"^\s*#+\s*", "", text).strip()

    @classmethod
    def normalize_heading_display(cls, text: str) -> str:
        value = cls.collapse_cjk_spaces(cls.strip_heading_markup(text))
        value = cls.SPACE_RE.sub(" ", value)
        return value.strip(" ：:;；,.。")

    @classmethod
    def normalize_heading_number(cls, text: str) -> str:
        return text.translate(cls.FULLWIDTH_HEADING_TRANSLATION)

    @classmethod
    def looks_like_section_heading(cls, line: str) -> str | None:
        for section, patterns in cls.SECTION_HEADING_PATTERNS.items():
            if any(pattern.fullmatch(line.strip()) for pattern in patterns):
                return section
        return None

    @classmethod
    def is_probable_body_heading_title(cls, title: str) -> bool:
        cleaned = cls.normalize_heading_display(title)
        if not cleaned or len(cleaned) > 60:
            return False
        if cls.BODY_REJECT_RE.match(cleaned) or re.search(r"[=<>$]", cleaned):
            return False
        return not cleaned.endswith(("。", "！", "？", "；", ";", "，", ","))

    @classmethod
    def parse_body_heading(cls, paragraph: str) -> HeadingInfo | None:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            return None
        first = cls.normalize_heading_display(lines[0])
        remainder = "\n".join(lines[1:]).strip()
        for kind, pattern in cls.BODY_HEADING_PATTERNS:
            match = pattern.fullmatch(first)
            if not match:
                continue
            label = cls.normalize_heading_display(match.group("label") or "")
            title = cls.normalize_heading_display(match.group("title") or "")
            if not label or not title or not cls.is_probable_body_heading_title(title):
                return None
            display = f"{label} {title}".strip() if kind == "chapter" else f"{label}{title}".strip()
            level = 1 if kind == "chapter" else 2 + cls.normalize_heading_number(label).count(".") if kind == "hierarchical" else 2
            return HeadingInfo(
                heading_level=level,
                heading_title=title,
                heading_display=display,
                remainder=remainder,
            )
        return None

    @classmethod
    def first_meaningful_anchor(cls, lines: list[str]) -> int:
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if cls.looks_like_section_heading(stripped) or cls.parse_body_heading(stripped):
                return index
        return -1

    def clean(self, raw_text: str) -> tuple[str, str]:
        lines = [self.normalize_line(line) for line in raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        title = self._detect_title(lines)
        lines = self._trim_front_matter(lines)
        cleaned_lines = self._clean_lines(lines)
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip() + "\n"
        return text, title

    def _detect_title(self, lines: list[str]) -> str:
        for line in lines:
            match = self.TITLE_RE.match(line.strip())
            if match:
                return self.normalize_heading_display(match.group(1))
        return ""

    def _trim_front_matter(self, lines: list[str]) -> list[str]:
        anchor = self.first_meaningful_anchor(lines)
        if anchor > 0:
            prefix = "\n".join(lines[:anchor]).strip()
            if self.FRONT_MATTER_SIGNAL_RE.search(prefix):
                return lines[anchor:]
        return lines

    def _clean_lines(self, lines: list[str]) -> list[str]:
        cleaned: list[str] = []
        in_toc = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if cleaned and cleaned[-1] != "":
                    cleaned.append("")
                continue
            if self.STOP_HEADING_RE.match(stripped):
                break
            if self.METADATA_LINE_RE.match(stripped):
                continue
            if re.fullmatch(r"(?:#+\s*)?(目录|Contents)", stripped, re.IGNORECASE):
                in_toc = True
                continue
            if in_toc:
                if self.TOC_LINE_RE.match(stripped):
                    continue
                if self.looks_like_section_heading(stripped) or self.parse_body_heading(stripped):
                    in_toc = False
                else:
                    continue
            cleaned.append(self.collapse_cjk_spaces(stripped))

        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        return cleaned
