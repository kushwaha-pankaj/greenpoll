#!/usr/bin/env python3
"""Convert CrossPoll LaTeX paper to PDF using reportlab"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import re
from pathlib import Path

# Configuration
PAPER_DIR = Path("paper/arxiv")
OUTPUT_PDF = Path("paper/CrossPoll_Paper.pdf")
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 0.75 * inch

def read_section(filename):
    """Read a LaTeX section file"""
    filepath = PAPER_DIR / "sections" / filename
    return filepath.read_text(encoding='utf-8')

def clean_latex_text(text):
    """Remove LaTeX commands and clean text"""
    # Remove labels
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    
    # Remove begin/end blocks
    text = re.sub(r'\\begin\{[^}]+\}', '', text)
    text = re.sub(r'\\end\{[^}]+\}', '', text)
    
    # Remove itemize markers
    text = re.sub(r'\\item\s+', '• ', text)
    
    # Handle bold text
    text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
    
    # Handle italic text
    text = re.sub(r'\\textit\{([^}]+)\}', r'\1', text)
    
    # Remove math mode
    text = re.sub(r'\$[^$]+\$', '[equation]', text)
    text = re.sub(r'\$\$[^$]+\$\$', '[equation block]', text)
    
    # Remove references
    text = re.sub(r'~\\ref\{[^}]+\}', '', text)
    text = re.sub(r'\\ref\{[^}]+\}', '', text)
    
    # Remove citation formatting
    text = re.sub(r'\\cite\{[^}]+\}', '[Ref]', text)
    
    # Remove checkmarks
    text = text.replace('\\checkmark', '✓')
    
    # Remove special characters
    text = text.replace('\\_', '_')
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def parse_latex_to_elements(latex_text, styles):
    """Parse LaTeX text into reportlab elements"""
    elements = []
    lines = latex_text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
            
        # Skip LaTeX commands
        if line.startswith('%') or line.startswith('\\'):
            continue
        
        # Handle subsections
        if '\\subsection{' in line:
            title = re.search(r'\\subsection\{([^}]+)\}', line)
            if title:
                elements.append(Paragraph(title.group(1), styles['Heading2']))
                elements.append(Spacer(1, 0.1*inch))
        
        # Handle subsubsections
        elif '\\subsubsection{' in line:
            title = re.search(r'\\subsubsection\{([^}]+)\}', line)
            if title:
                elements.append(Paragraph(title.group(1), styles['Heading3']))
                elements.append(Spacer(1, 0.05*inch))
        
        # Regular text
        elif line and not line.startswith('\\'):
            cleaned = clean_latex_text(line)
            if cleaned and cleaned != '[equation]':
                elements.append(Paragraph(cleaned, styles['BodyText']))
                elements.append(Spacer(1, 0.05*inch))
    
    return elements

def create_pdf():
    """Create the PDF document"""
    # Create styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.black,
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.black,
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#2e5c8a'),
        spaceAfter=6,
        spaceBefore=6,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    # Override default styles
    custom_styles = {
        'Title': title_style,
        'Heading1': heading1_style,
        'Heading2': heading2_style,
        'Heading3': heading3_style,
        'BodyText': body_style
    }
    
    # Document elements
    elements = []
    
    # Title page
    elements.append(Paragraph(
        'CrossPoll: Sample-Efficient Cross-Crop Flower Detection via Multi-Source Transfer Learning for Robotic Pollination',
        custom_styles['Title']
    ))
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph('Pankaj Kushwaha', custom_styles['BodyText']))
    elements.append(Paragraph('Department of Computer Science and Engineering', custom_styles['BodyText']))
    elements.append(Spacer(1, 0.2*inch))
    from datetime import datetime
    elements.append(Paragraph(datetime.now().strftime('%B %d, %Y'), custom_styles['BodyText']))
    elements.append(PageBreak())
    
    # Sections
    sections = [
        ('abstract.tex', 'Abstract'),
        ('introduction.tex', '1. Introduction'),
        ('related_work.tex', '2. Related Work'),
        ('methods.tex', '3. Methodology'),
        ('experiments.tex', '4. Experimental Setup'),
        ('results.tex', '5. Results'),
        ('discussion.tex', '6. Discussion'),
        ('conclusion.tex', '7. Conclusion'),
    ]
    
    for filename, section_title in sections:
        # Add section heading
        elements.append(Paragraph(section_title, custom_styles['Heading1']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Read and parse section
        try:
            section_text = read_section(filename)
            section_elements = parse_latex_to_elements(section_text, custom_styles)
            elements.extend(section_elements)
        except FileNotFoundError:
            elements.append(Paragraph(f'[Section file {filename} not found]', custom_styles['BodyText']))
        
        # Page break between sections (except last)
        if section_title != '7. Conclusion':
            elements.append(PageBreak())
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )
    
    # Build PDF
    doc.build(elements)
    print(f'✅ PDF document created: {OUTPUT_PDF}')

if __name__ == '__main__':
    create_pdf()
