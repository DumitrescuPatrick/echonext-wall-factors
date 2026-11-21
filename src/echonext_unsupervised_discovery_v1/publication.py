"""Export an aggregate-only draft in Word/PDF and an explicit allowlist ZIP."""
import argparse,hashlib,json,re,textwrap,zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def documents(source,destination):
    text=source.read_text(); paragraphs=[re.sub(r'\*\*(.*?)\*\*',r'\1',p.replace('\n',' ')) for p in text.split('\n\n') if p.strip()]
    body=[]
    for p in paragraphs:
        level=len(p)-len(p.lstrip('#'));content=p.lstrip('# ');style=f'<w:pPr><w:pStyle w:val="Heading{min(level,2)}"/></w:pPr>' if level else ''
        body.append(f'<w:p>{style}<w:r><w:t xml:space="preserve">{escape(content)}</w:t></w:r></w:p>')
    document='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+''.join(body)+'<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body></w:document>'
    with zipfile.ZipFile(destination/'MANUSCRIPT_DRAFT.docx','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr('word/_rels/document.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        z.writestr('word/document.xml',document)
        z.writestr('word/styles.xml','<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:rPrDefault></w:docDefaults><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style></w:styles>')
    with PdfPages(destination/'MANUSCRIPT_DRAFT.pdf') as pdf:
        fig=None;y=0;page=0
        for p in paragraphs:
            heading=p.startswith('#');p=p.lstrip('# ');lines=textwrap.wrap(p,85 if heading else 100,break_long_words=True)
            for i,line in enumerate(lines):
                if fig is None or y<.09:
                    if fig is not None:pdf.savefig(fig);plt.close(fig)
                    fig=plt.figure(figsize=(8.27,11.69));page+=1;y=.94;fig.text(.5,.04,f'Exploratory research draft | {page}',ha='center',fontsize=8,color='.35')
                fig.text(.08,y,line,fontsize=11 if heading else 10,fontweight='bold' if heading else 'normal',family='DejaVu Serif');y-=.019 if heading else .017
            y-=.012
        if fig is not None:pdf.savefig(fig);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));a=p.parse_args();root=a.root;destination=root/'reports/unsupervised_discovery_v1';documents(root/'docs/unsupervised_discovery_v1/MANUSCRIPT_DRAFT.md',destination)
    files=[]
    for directory in ['src/echonext_unsupervised_discovery_v1','tests/unsupervised_discovery_v1','docs/unsupervised_discovery_v1','config/unsupervised_discovery_v1','reports/unsupervised_discovery_v1']:
        files.extend(f for f in (root/directory).rglob('*') if f.is_file() and f.suffix in {'.py','.md','.json','.csv','.png','.pdf','.svg','.docx','.txt'})
    files.extend(root/'src/echonext'/name for name in ['__init__.py','data.py','schema.py','stage2_models.py','stage2_cache.py'])
    files.extend(root/'reports/stage2_v1'/name for name in ['development_cache_integrity.json','cache_bitwise_verification.json'])
    # Only explicitly allowed new aggregate paths + minimal frozen read-only dependencies.
    for f in files:
        if 'private' in f.relative_to(root).parts or f.suffix in {'.npy','.npz','.pt','.pth','.pkl','.joblib'}:raise ValueError('Private artifact in public manifest')
    manifest={'publication_status':'Exploratory draft; neural clinical experiments blocked by unavailable GPU','files':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(files))}}
    manifest_path=destination/'PACKAGE_MANIFEST.json';manifest['files'].pop(str(manifest_path.relative_to(root)),None);manifest_path.write_text(json.dumps(manifest,indent=2))
    archive=root/'EchoNext_Unsupervised_Discovery_v1_Aggregate_Package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(set(files+[manifest_path])):z.write(f,str(f.relative_to(root)))
    print(f'Aggregate-only draft package: {archive.name}; {len(set(files+[manifest_path]))} files')
if __name__=='__main__':main()
