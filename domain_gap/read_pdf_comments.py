"""
Extract review comments (annotations) from a PDF, including the text each
highlight covers.

Usage:  python read_pdf_comments.py /path/to/AI_sorting.pdf

Acrobat stores a highlight as coordinates, not as the selected words, so the
highlighted text is recovered by intersecting each annotation's /QuadPoints
with the word boxes reported by `pdftotext -bbox`.

NOTE: Acrobat keeps comments in memory until the file is saved -- the reviewer
must save the PDF (Cmd+S) before the comments appear here.
"""
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from pypdf import PdfReader

MARKUP = {"Text", "Highlight", "StrikeOut", "Underline", "Squiggly",
          "FreeText", "Ink", "Square", "Circle", "Caret", "Stamp",
          "Polygon", "PolyLine", "FileAttachment"}


def _pdftotext():
    exe = shutil.which("pdftotext")
    if exe:
        return exe
    for cand in ("/opt/homebrew/bin/pdftotext", "/usr/local/bin/pdftotext",
                 "/Library/TeX/texbin/pdftotext"):
        if Path(cand).exists():
            return cand
    raise RuntimeError("pdftotext not found (install poppler: brew install poppler)")


PDFTOTEXT = _pdftotext()


def word_boxes(path):
    """{page_no: [(x0, y0, x1, y1, word)]} in top-left origin coords."""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        out = tmp.name
    subprocess.run([PDFTOTEXT, "-bbox", str(path), out],
                   check=True, capture_output=True)
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    root = ET.parse(out).getroot()
    pages = {}
    for pno, page in enumerate(root.iter(f"{{{ns['x']}}}page"), start=1):
        words = []
        for w in page.iter(f"{{{ns['x']}}}word"):
            words.append((float(w.get("xMin")), float(w.get("yMin")),
                          float(w.get("xMax")), float(w.get("yMax")),
                          (w.text or "")))
        pages[pno] = words
    Path(out).unlink(missing_ok=True)
    return pages


def quad_rects(obj, page_height):
    """/QuadPoints -> list of (x0, y0, x1, y1) in top-left origin coords."""
    qp = obj.get("/QuadPoints")
    if not qp:
        r = obj.get("/Rect")
        if not r:
            return []
        x0, y0, x1, y1 = [float(v) for v in r]
        return [(min(x0, x1), page_height - max(y0, y1),
                 max(x0, x1), page_height - min(y0, y1))]
    vals = [float(v) for v in qp]
    rects = []
    for i in range(0, len(vals), 8):
        xs = vals[i:i + 8:2]
        ys = vals[i + 1:i + 8:2]
        if len(xs) < 4:
            continue
        rects.append((min(xs), page_height - max(ys),
                      max(xs), page_height - min(ys)))
    return rects


def text_under(rects, words, pad=1.0):
    """Words whose centre falls inside any highlight rect."""
    picked = []
    for (x0, y0, x1, y1) in rects:
        for (wx0, wy0, wx1, wy1, txt) in words:
            cx, cy = (wx0 + wx1) / 2, (wy0 + wy1) / 2
            if x0 - pad <= cx <= x1 + pad and y0 - pad <= cy <= y1 + pad:
                picked.append(txt)
    return " ".join(picked).strip()


def extract(path):
    reader = PdfReader(path)
    boxes = word_boxes(path)
    out = []
    for pno, page in enumerate(reader.pages, start=1):
        ph = float(page.mediabox.height)
        for annot in (page.get("/Annots") or []):
            try:
                obj = annot.get_object()
            except Exception:
                continue
            sub = str(obj.get("/Subtype", "")).lstrip("/")
            if sub not in MARKUP:
                continue  # skip hyperref /Link
            quoted = text_under(quad_rects(obj, ph), boxes.get(pno, []))
            out.append({
                "page": pno,
                "type": sub,
                "author": str(obj.get("/T", "") or ""),
                "body": str(obj.get("/Contents", "") or "").strip(),
                "quoted": quoted,
            })
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    p = Path(sys.argv[1])
    ann = extract(p)
    if not ann:
        print(f"No review comments found in {p.name}. Was the PDF saved?")
        return
    print(f"{len(ann)} comment(s) in {p.name}\n" + "=" * 70)
    for i, a in enumerate(sorted(ann, key=lambda x: x["page"]), start=1):
        who = f" by {a['author']}" if a["author"] else ""
        print(f"\n--- [{i}] p.{a['page']} {a['type']}{who} ---")
        if a["quoted"]:
            print(f"  HIGHLIGHTED: “{a['quoted']}”")
        print(f"  COMMENT: {a['body'] or '(markup only, no text)'}")


if __name__ == "__main__":
    main()
