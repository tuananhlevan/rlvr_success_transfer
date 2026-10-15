"""Check Markdown links and compile report math in a temporary directory.

Writes one validation JSON. No HTML, LaTeX build products or logs are retained.
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/validation_results.json'
files=sorted([*ROOT.glob('reports/**/*.md'),*ROOT.glob('studies/**/*.md'),ROOT/'resources/README.md'])
pattern=re.compile(r'(?<!\\)\$\$(.+?)(?<!\\)\$\$|(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)',re.S)
expressions=[];counts={};broken=[]
for p in files:
    body=p.read_text();body=re.sub(r'```.*?```','',body,flags=re.S)
    matches=list(pattern.finditer(body));counts[str(p.relative_to(ROOT))]=len(matches)
    assert not re.search(r'(?<!\\)\$',pattern.sub('',body)),f'Unmatched math delimiter: {p}'
    expressions.extend(m.group(1) or m.group(2) for m in matches)
    for target in re.findall(r'\]\(([^)]+)\)',body):
        if target.startswith(('http:','https:','#','mailto:')):continue
        local=unquote(target.strip('<>').split('#')[0])
        resolved=(p.parent/local).resolve()
        if resolved!=OUTPUT and not resolved.exists():broken.append({'file':str(p.relative_to(ROOT)),'target':target})
assert not broken,broken
with tempfile.TemporaryDirectory(prefix='research-math-') as temporary:
    temp=Path(temporary)
    chunks=[r'\documentclass{article}',r'\usepackage{amsmath,amssymb}',r'\begin{document}']
    for expression in expressions:
        chunks.extend([r'\[',re.sub(r'\n\s*\n','\n',expression.strip()),r'\]'])
    chunks.append(r'\end{document}')
    tex=temp/'math.tex';tex.write_text('\n'.join(chunks)+'\n')
    run=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',str(temp),str(tex)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout[-5000:]+run.stderr
result={'expression_counts':counts,'total_expressions':len(expressions),'latex_exit_code':0,
        'broken_local_links':broken,'temporary_build_files_retained':False,'html_generated':False}
OUTPUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
