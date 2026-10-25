#!/usr/bin/env python3
"""Export selected project source files as a polished, syntax-highlighted PDF."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = ["hermes_app.py"]
DEFAULT_OUTPUT = PROJECT_ROOT / "HERMES_code_showcase.pdf"
HTML_OUTPUT = PROJECT_ROOT / "HERMES_code_showcase.html"
LINES_PER_BLOCK = 52
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]
SECRET_ASSIGNMENT = re.compile(
    r'(?P<prefix>\b(?:databasepassword|password|passwd|secret|token|api_key)\b\s*=\s*)(?P<quote>["\']).*?(?P=quote)',
    re.IGNORECASE,
)


CSS = """
@page {
  size: A4;
  margin: 12mm;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 1mm;
  color: #0f172a;
  background: #ffffff;
  font-family: "SFMono-Regular", "Cascadia Code", "JetBrains Mono", "Menlo", "Consolas", monospace;
  font-size: 10pt;
  line-height: 1.42;
}

.file {
  margin: 0;
}

.code-page {
  break-after: page;
  page-break-after: always;
}

.code-page:last-child {
  break-after: auto;
  page-break-after: auto;
}

.code-wrap {
  overflow: visible;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  background: #ffffff;
  padding: 1px;
  break-inside: avoid;
  page-break-inside: avoid;
}

table.highlighttable {
  width: 100%;
  border-spacing: 0;
  border-collapse: collapse;
}

td.linenos {
  width: 12mm;
  padding: 4mm 2.5mm;
  background: #f8fafc;
  color: #64748b;
  text-align: right;
  user-select: none;
  vertical-align: top;
}

td.code {
  padding: 4mm;
  vertical-align: top;
}

pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: "SFMono-Regular", "Cascadia Code", "JetBrains Mono", "Menlo", "Consolas", monospace;
  font-size: 9.4pt;
  line-height: 1.42;
}

.hll { background-color: #fff6c7 }
.c, .ch, .cm, .cp, .cpf, .c1, .cs { color: #6a737d; font-style: italic }
.k, .kc, .kd, .kn, .kp, .kr, .kt { color: #a626a4; font-weight: 700 }
.o, .ow { color: #db2777; font-weight: 700 }
.m, .mb, .mf, .mh, .mi, .mo, .il { color: #986801 }
.s, .sa, .sb, .sc, .dl, .sd, .s2, .se, .sh, .si, .sx, .sr, .s1, .ss { color: #50a14f }
.n, .nv, .vc, .vg, .vi, .vm { color: #0f172a }
.na { color: #0f766e }
.nc, .ne { color: #b45309; font-weight: 700 }
.nd, .nf, .fm { color: #0891b2; font-weight: 700 }
.nn { color: #0f172a }
.nb, .bp { color: #dc2626 }
.p { color: #475569 }
.gh, .gu { color: #182032; font-weight: 800 }
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a polished PDF/HTML code showcase for HERMES."
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=DEFAULT_FILES,
        help="Source files to include. Defaults to hermes_app.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="PDF output path.",
    )
    parser.add_argument(
        "--title",
        default="Project H.E.R.M.E.S.",
        help="Cover title.",
    )
    parser.add_argument(
        "--subtitle",
        default="Hand-sign Extraction and Recognition using Machine-learning for Enhanced Speech",
        help="Cover subtitle.",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Only write the HTML preview.",
    )
    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="Include obvious password/token assignments without redaction.",
    )
    return parser.parse_args()


def read_file(path_text: str) -> tuple[Path, str]:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path, path.read_text(encoding="utf-8")


def redact_secrets(code: str) -> str:
    return SECRET_ASSIGNMENT.sub(r'\g<prefix>\g<quote>***REDACTED***\g<quote>', code)


def highlight_code(filename: str, code: str, linenostart: int = 1) -> str:
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_for_filename
    except ImportError:
        escaped = html.escape(code)
        lines = "\n".join(
            str(index) for index, _ in enumerate(code.splitlines(), linenostart)
        )
        return (
            '<table class="highlighttable"><tr>'
            f'<td class="linenos"><pre>{lines}</pre></td>'
            f'<td class="code"><pre>{escaped}</pre></td>'
            "</tr></table>"
        )

    lexer = get_lexer_for_filename(filename, stripnl=False)
    formatter = HtmlFormatter(
        cssclass="highlight",
        linenos="table",
        linenostart=linenostart,
        nowrap=False,
        style="default",
    )
    return highlight(code, lexer, formatter)


def build_html(files: list[tuple[Path, str]], title: str, subtitle: str) -> str:
    sections = []
    for path, code in files:
        lines = code.splitlines()
        if code.endswith("\n"):
            lines.append("")
        for index in range(0, len(lines), LINES_PER_BLOCK):
            chunk = "\n".join(lines[index : index + LINES_PER_BLOCK])
            if chunk:
                chunk += "\n"
            sections.append(
                '<section class="code-page">'
                f'<div class="code-wrap">{highlight_code(path.name, chunk, index + 1)}</div>'
                "</section>"
            )
    file_sections = "\n".join(sections)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)} Code Showcase</title>
  <style>{CSS}</style>
</head>
<body>
  {file_sections}
</body>
</html>
"""


def find_browser() -> Path | None:
    for browser in CHROME_CANDIDATES:
        path = Path(browser)
        if path.exists():
            return path
    return None


def write_pdf_with_browser(output: Path) -> bool:
    browser = find_browser()
    if browser is None:
        return False

    command = [
        str(browser),
        "--headless",
        "--disable-gpu",
        "--no-first-run",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output}",
        str(HTML_OUTPUT.resolve().as_uri()),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True


def write_pdf_with_weasyprint(output: Path) -> bool:
    try:
        from weasyprint import HTML
    except (ImportError, OSError):
        return False

    HTML(filename=str(HTML_OUTPUT)).write_pdf(str(output))
    return True


def main() -> int:
    args = parse_args()
    try:
        files = [read_file(path_text) for path_text in args.files]
    except OSError as exc:
        print(f"Could not read source file: {exc}", file=sys.stderr)
        return 1

    if not args.no_redact:
        files = [(path, redact_secrets(code)) for path, code in files]

    html_text = build_html(files, args.title, args.subtitle)
    HTML_OUTPUT.write_text(html_text, encoding="utf-8")
    print(f"Wrote HTML preview: {HTML_OUTPUT}")

    if args.html_only:
        return 0

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    try:
        wrote_pdf = write_pdf_with_browser(output) or write_pdf_with_weasyprint(output)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"PDF export failed: {exc}", file=sys.stderr)
        return 1

    if not wrote_pdf:
        print(
            "Could not find Chrome/Brave/Edge, and WeasyPrint is not fully available.",
            file=sys.stderr,
        )
        print("Open the HTML preview and print it to PDF from your browser.", file=sys.stderr)
        return 2

    print(f"Wrote PDF: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
