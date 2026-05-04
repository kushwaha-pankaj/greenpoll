#!/usr/bin/env node

const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, HeadingLevel, 
        AlignmentType, PageBreak, BorderStyle, WidthType, ShadingType, VerticalAlign,
        UnderlineType, PageNumber } = require('docx');
const fs = require('fs');
const path = require('path');

// Read LaTeX section files
const sectionsDir = path.join(__dirname, 'paper/arxiv/sections');

function readSection(filename) {
  const filepath = path.join(sectionsDir, filename);
  return fs.readFileSync(filepath, 'utf8');
}

// Parse LaTeX into structured content
function parseLatexToDocx(latexText) {
  const paragraphs = [];
  const lines = latexText.split('\n');
  let currentText = '';

  for (const line of lines) {
    // Skip section labels
    if (line.includes('\\label{')) continue;
    
    // Skip begin/end enumerate, itemize, etc.
    if (line.includes('\\begin{') || line.includes('\\end{')) continue;
    
    // Handle subsections
    if (line.includes('\\subsection{')) {
      if (currentText.trim()) {
        paragraphs.push(new Paragraph({
          text: currentText.trim(),
          style: "Normal"
        }));
        currentText = '';
      }
      const title = line.replace(/\\subsection\{/, '').replace(/\}/, '');
      paragraphs.push(new Paragraph({
        text: title,
        heading: HeadingLevel.HEADING_2,
        bold: true,
        spacing: { before: 240, after: 120 }
      }));
    }
    // Handle subsubsections
    else if (line.includes('\\subsubsection{')) {
      if (currentText.trim()) {
        paragraphs.push(new Paragraph({
          text: currentText.trim(),
          style: "Normal"
        }));
        currentText = '';
      }
      const title = line.replace(/\\subsubsection\{/, '').replace(/\}/, '');
      paragraphs.push(new Paragraph({
        text: title,
        heading: HeadingLevel.HEADING_3,
        bold: true,
        spacing: { before: 120, after: 80 }
      }));
    }
    // Handle enumerate/itemize items
    else if (line.includes('\\item ')) {
      if (currentText.trim()) {
        paragraphs.push(new Paragraph({
          text: currentText.trim(),
          style: "Normal"
        }));
        currentText = '';
      }
      const itemText = line.replace(/\s*\\item\s+/, '');
      paragraphs.push(new Paragraph({
        text: itemText.trim(),
        bullet: { level: 0 },
        spacing: { after: 80 }
      }));
    }
    // Handle bold \textbf{}
    else if (line.includes('\\textbf{')) {
      const matches = line.match(/\\textbf\{([^}]+)\}/);
      if (matches) {
        const boldText = matches[1];
        const beforeText = line.substring(0, line.indexOf('\\textbf{'));
        const afterText = line.substring(line.indexOf('}', line.indexOf('\\textbf{')) + 1);
        
        paragraphs.push(new Paragraph({
          children: [
            new TextRun({ text: beforeText, bold: false }),
            new TextRun({ text: boldText, bold: true }),
            new TextRun({ text: afterText, bold: false })
          ],
          spacing: { after: 80 }
        }));
      }
    }
    // Regular text lines
    else if (line.trim()) {
      currentText += ' ' + line.trim();
    }
  }

  if (currentText.trim()) {
    paragraphs.push(new Paragraph({
      text: currentText.trim(),
      style: "Normal"
    }));
  }

  return paragraphs;
}

// Create document sections
async function createDocument() {
  const sections = [];

  // Title page
  sections.push(
    new Paragraph({
      text: 'CrossPoll: Sample-Efficient Cross-Crop Flower Detection via Multi-Source Transfer Learning for Robotic Pollination',
      heading: HeadingLevel.HEADING_1,
      bold: true,
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 }
    }),
    new Paragraph({
      text: 'Pankaj Kushwaha',
      alignment: AlignmentType.CENTER,
      spacing: { after: 80 }
    }),
    new Paragraph({
      text: 'Department of Computer Science and Engineering',
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 }
    }),
    new Paragraph({
      text: new Date().toLocaleDateString(),
      alignment: AlignmentType.CENTER,
      spacing: { after: 600 }
    }),
    new PageBreak()
  );

  // Abstract
  const abstractText = readSection('abstract.tex');
  sections.push(
    new Paragraph({
      text: 'Abstract',
      heading: HeadingLevel.HEADING_1,
      bold: true,
      spacing: { before: 200, after: 120 }
    }),
    ...parseLatexToDocx(abstractText),
    new PageBreak()
  );

  // Introduction
  const introText = readSection('introduction.tex');
  sections.push(
    new Paragraph({
      text: '1. Introduction',
      heading: HeadingLevel.HEADING_1,
      bold: true,
      spacing: { before: 200, after: 120 }
    }),
    ...parseLatexToDocx(introText),
    new PageBreak()
  );

  // Related Work
  const relatedText = readSection('related_work.tex');
  sections.push(
    new Paragraph({
      text: '2. Related Work',
      heading: HeadingLevel.HEADING_1,
      bold: true,
      spacing: { before: 200, after: 120 }
    }),
    ...parseLatexToDocx(relatedText),
    new PageBreak()
  );

  // Methodology
  const methodsText = readSection('methods.tex');
  sections.push(
    new Paragraph({
      text: '3. Methodology',
      heading: HeadingLevel.HEADING_1,
      bold: true,
      spacing: { before: 200, after: 120 }
    }),
    ...parseLatexToDocx(methodsText),
    new PageBreak()
  );

  // Experiments
  const experimentsText = readSection('experiments.tex');
  sections.push(
    new Paragraph({
      text: '4. Experimental Setup',
      heading: HeadingLevel.HEADING_1,
      bold: true,
      spacing: { before: 200, after: 120 }
    }),
    ...parseLatexToDocx(experimentsText),
    new PageBreak()
  );

  // Results
  const resultsText = readSection('results.tex');
  sections.push(
    new Paragraph({
      text: '5. Results',
      heading: HeadingLevel.HEADING_1,
      bold: true,
      spacing: { before: 200, after: 120 }
    }),
    ...parseLatexToDocx(resultsText),
    new PageBreak()
  );

  // Discussion
  const discussionText = readSection('discussion.tex');
  sections.push(
    new Paragraph({
      text: '6. Discussion',
      heading: HeadingLevel.HEADING_1,
      bold: true,
      spacing: { before: 200, after: 120 }
    }),
    ...parseLatexToDocx(discussionText),
    new PageBreak()
  );

  // Conclusion
  const conclusionText = readSection('conclusion.tex');
  sections.push(
    new Paragraph({
      text: '7. Conclusion',
      heading: HeadingLevel.HEADING_1,
      bold: true,
      spacing: { before: 200, after: 120 }
    }),
    ...parseLatexToDocx(conclusionText)
  );

  // Create document
  const doc = new Document({
    sections: [{
      properties: {
        page: {
          margin: {
            top: 1440,    // 1 inch
            right: 1440,
            bottom: 1440,
            left: 1440
          }
        }
      },
      children: sections
    }]
  });

  // Save document
  const buffer = await Packer.toBuffer(doc);
  const outputPath = path.join(__dirname, 'paper/CrossPoll_Paper.docx');
  fs.writeFileSync(outputPath, buffer);

  console.log(`✅ Word document created: ${outputPath}`);
  return outputPath;
}

// Run
createDocument().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
