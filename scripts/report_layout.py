"""ReportLab layout shared by the assignment reports (A4, DejaVu Sans, header "name | course | topic").

Used by scripts/build_a2c_report.py; keeps page styling apart from the report text.
"""
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.fonts import addMapping
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageTemplate, Paragraph, Table, TableStyle

_FONTS = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
for name, file in (('DejaVu', 'DejaVuSans.ttf'), ('DejaVu-Bold', 'DejaVuSans-Bold.ttf'),
                   ('DejaVu-Oblique', 'DejaVuSans-Oblique.ttf'),
                   ('DejaVu-BoldOblique', 'DejaVuSans-BoldOblique.ttf')):
    pdfmetrics.registerFont(TTFont(name, str(_FONTS / file)))
addMapping('DejaVu', 0, 0, 'DejaVu')
addMapping('DejaVu', 1, 0, 'DejaVu-Bold')
addMapping('DejaVu', 0, 1, 'DejaVu-Oblique')
addMapping('DejaVu', 1, 1, 'DejaVu-BoldOblique')

INK = colors.HexColor('#14293d')
MUTED = colors.HexColor('#4a5d70')
RULE = colors.HexColor('#c9d2db')
SHADE = colors.HexColor('#f1f4f7')
LINK = '#1f5fbf'

PAGE_W, PAGE_H = A4
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 40, 44, 30
CONTENT_W = PAGE_W - 2 * MARGIN_X


def styles(body_size=8.6, leading=10.9):
    base = dict(fontName='DejaVu', textColor=INK)
    return {
        'title': ParagraphStyle('title', fontSize=19, leading=23, spaceAfter=2, **base),
        'byline': ParagraphStyle('byline', fontSize=body_size, leading=leading + 1, textColor=MUTED,
                                 fontName='DejaVu', spaceAfter=4),
        'h2': ParagraphStyle('h2', fontSize=12.5, leading=15, spaceBefore=4, spaceAfter=2,
                             fontName='DejaVu-Bold', textColor=INK),
        'body': ParagraphStyle('body', fontSize=body_size, leading=leading, alignment=TA_JUSTIFY,
                               spaceAfter=2, **base),
        'small': ParagraphStyle('small', fontSize=6.7, leading=8.0, textColor=MUTED, fontName='DejaVu',
                                alignment=TA_JUSTIFY, spaceAfter=2),
        'cell': ParagraphStyle('cell', fontSize=7.0, leading=8.2, **base),
        'cellc': ParagraphStyle('cellc', fontSize=7.0, leading=8.2, alignment=TA_CENTER, **base),
        'cellh': ParagraphStyle('cellh', fontSize=7.0, leading=8.2, alignment=TA_CENTER,
                                fontName='DejaVu-Bold', textColor=INK),
    }


def link(url, text=None):
    return f'<a href="{url}" color="{LINK}">{text or url}</a>'


def figure(path, width=CONTENT_W):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def table(rows, col_widths, st, header_rows=1):
    """rows: list of lists of strings (first row = header)."""
    data = [[Paragraph(c, st['cellh'] if r < header_rows else (st['cell'] if j == 0 else st['cellc']))
             for j, c in enumerate(row)] for r, row in enumerate(rows)]
    t = Table(data, colWidths=col_widths, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 0.8, INK),
        ('LINEBELOW', (0, header_rows - 1), (-1, header_rows - 1), 0.5, INK),
        ('LINEBELOW', (0, -1), (-1, -1), 0.8, INK),
        ('ROWBACKGROUNDS', (0, header_rows), (-1, -1), [colors.white, SHADE]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
    ]))
    return t


class _NumberedCanvas(canvas.Canvas):
    """Draws the running header with 'page/total' once the page count is known."""
    header_text = ''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._draw_header(total)
            super().showPage()
        super().save()

    def _draw_header(self, total):
        self.saveState()
        self.setFont('DejaVu', 7)
        self.setFillColor(MUTED)
        y = PAGE_H - 26
        self.drawString(MARGIN_X, y, self.header_text)
        self.drawRightString(PAGE_W - MARGIN_X, y, f'{self._pageNumber}/{total}')
        self.setStrokeColor(RULE)
        self.setLineWidth(0.5)
        self.line(MARGIN_X, y - 4, PAGE_W - MARGIN_X, y - 4)
        self.restoreState()


def build(path, story, *, header, title, author, subject, keywords):
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
                          topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
                          title=title, author=author, subject=subject, keywords=keywords,
                          creator='scripts/build_a2c_report.py')
    frame = Frame(MARGIN_X, MARGIN_BOTTOM, CONTENT_W, PAGE_H - MARGIN_TOP - MARGIN_BOTTOM,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='body')
    doc.addPageTemplates([PageTemplate(id='page', frames=[frame])])
    canvas_cls = type('HeaderCanvas', (_NumberedCanvas,), {'header_text': header})
    doc.build(story, canvasmaker=canvas_cls)
    return doc
