from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

md_path = 'EXPLICACION_02_asl.md'
pdf_path = 'EXPLICACION_02_asl.pdf'

# Simple markdown -> reportlab paragraph converter
def md_to_flowables(md_text):
    styles = getSampleStyleSheet()
    normal = styles['BodyText']
    h1 = ParagraphStyle('h1', parent=styles['Heading1'], spaceAfter=12)
    h2 = ParagraphStyle('h2', parent=styles['Heading2'], spaceAfter=8)
    code = ParagraphStyle('code', parent=styles['Code'], fontName='Courier', fontSize=8, spaceAfter=6)

    flow = []
    lines = md_text.splitlines()
    buffer = []

    def flush_para():
        nonlocal buffer
        if not buffer:
            return
        text = ' '.join(buffer).strip()
        flow.append(Paragraph(text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), normal))
        flow.append(Spacer(1, 6))
        buffer = []

    in_code = False
    code_block = []

    for line in lines:
        if line.strip().startswith('```'):
            if in_code:
                # end code block
                flow.append(Paragraph('<br/>'.join(code_block).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), code))
                flow.append(Spacer(1,6))
                code_block = []
                in_code = False
            else:
                flush_para()
                in_code = True
            continue
        if in_code:
            code_block.append(line)
            continue
        if line.startswith('# '):
            flush_para()
            flow.append(Paragraph(line[2:].strip(), h1))
            continue
        if line.startswith('## '):
            flush_para()
            flow.append(Paragraph(line[3:].strip(), h2))
            continue
        if line.strip() == '---':
            flush_para()
            flow.append(Spacer(1,12))
            continue
        if line.strip() == '':
            flush_para()
            continue
        # normal text
        buffer.append(line.strip())
    flush_para()
    return flow


def main():
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=72)
    flowables = md_to_flowables(md_text)
    doc.build(flowables)
    print('PDF generado:', pdf_path)

if __name__ == '__main__':
    main()
