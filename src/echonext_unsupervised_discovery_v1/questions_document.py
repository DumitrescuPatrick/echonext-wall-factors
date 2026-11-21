"""Render the follow-up report with real Word/PDF tables; no patient data."""
import re,textwrap,zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path('reports/unsupervised_discovery_v1/five_questions_v1')
SOURCE=Path('docs/unsupervised_discovery_v1/five_questions_v1/FIVE_QUESTIONS_RESULTS.md')

def blocks(text):
    for part in text.split('\n\n'):
        if not part.strip():continue
        if part.lstrip().startswith('|'):
            rows=[[c.strip() for c in line.strip().strip('|').split('|')] for line in part.splitlines() if line.strip().startswith('|') and not re.match(r'^\|[\s:|\-]+\|$',line.strip())];yield 'table',rows
        else:yield 'text',re.sub(r'\*\*(.*?)\*\*',r'\1',part.replace('\n',' '))

def main():
    content=list(blocks(SOURCE.read_text()));body=[]
    for kind,value in content:
        if kind=='text':
            level=len(value)-len(value.lstrip('#'));style=f'<w:pPr><w:pStyle w:val="Heading{min(level,2)}"/></w:pPr>' if level else '';body.append('<w:p>'+style+'<w:r><w:t xml:space="preserve">'+escape(value.lstrip('# '))+'</w:t></w:r></w:p>')
        else:
            table='<w:tbl><w:tblPr><w:tblW w:w="9638" w:type="dxa"/><w:tblBorders>'+''.join(f'<w:{side} w:val="single" w:sz="4" w:color="CCCCCC"/>' for side in ['top','left','bottom','right','insideH','insideV'])+'</w:tblBorders></w:tblPr>'
            for i,row in enumerate(value):
                table+='<w:tr>'
                for cell in row:table+='<w:tc><w:p><w:r><w:rPr><w:sz w:val="20"/>'+('<w:b/>' if i==0 else '')+'</w:rPr><w:t>'+escape(cell)+'</w:t></w:r></w:p></w:tc>'
                table+='</w:tr>'
            body.append(table+'</w:tbl>')
    xml='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+''.join(body)+'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body></w:document>'
    docx=ROOT/'FIVE_QUESTIONS_RESULTS.docx'
    with zipfile.ZipFile(docx) as z:parts={n:z.read(n) for n in z.namelist()}
    parts['word/document.xml']=xml.encode()
    with zipfile.ZipFile(docx,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in parts.items():z.writestr(n,b)
    with PdfPages(ROOT/'FIVE_QUESTIONS_RESULTS.pdf') as pdf:
        fig=None;y=0;page=0
        def new_page():
            nonlocal fig,y,page
            if fig is not None:pdf.savefig(fig);plt.close(fig)
            fig=plt.figure(figsize=(8.27,11.69));page+=1;y=.94;fig.text(.5,.04,f'Exploratory EchoNext research report | {page}',ha='center',fontsize=8,color='.35')
        for kind,value in content:
            if kind=='text':
                heading=value.startswith('#');lines=textwrap.wrap(value.lstrip('# '),85 if heading else 100,break_long_words=True)
                if heading and (fig is None or y<.16):new_page()
                for line in lines:
                    if fig is None or y<.09:new_page()
                    fig.text(.08,y,line,fontsize=11 if heading else 10,fontweight='bold' if heading else 'normal',family='DejaVu Serif');y-=.019 if heading else .017
                y-=.012
            else:
                cols=len(value[0]);lengths=np_lengths=[min(48,max(len(row[j]) for row in value)) for j in range(cols)];weights=[v/sum(lengths) for v in lengths];wrapped=[[textwrap.fill(cell,max(12,int(100*weights[j])),break_long_words=False) for j,cell in enumerate(row)] for row in value];heights=[max(cell.count('\n')+1 for cell in row)*.017+.012 for row in wrapped];height=sum(heights)
                if fig is None or y-height<.10:new_page()
                axis=fig.add_axes([.08,y-height,.84,height]);axis.axis('off');table=axis.table(cellText=wrapped[1:],colLabels=wrapped[0],colWidths=weights,loc='center',bbox=[0,0,1,1],cellLoc='left');table.auto_set_font_size(False);table.set_fontsize(8.5)
                for (i,j),cell in table.get_celld().items():
                    cell.set_height(heights[i]/height);cell.set_edgecolor('#CCCCCC')
                    if i==0:cell.set_facecolor('#EAF0F4');cell.set_text_props(weight='bold')
                y-=height+.025
        if fig is not None:pdf.savefig(fig);plt.close(fig)
    print('Word/PDF report rendered with formatted tables.')
if __name__=='__main__':main()
