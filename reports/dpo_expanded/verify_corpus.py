"""Verify the browser-downloaded spreadsheet corpus and refresh its manifest.

No network or browser access. Existing PDFs are reused, never overwritten.
Run from any directory: python3 reports/dpo_expanded/verify_corpus.py
"""
import hashlib
import json
import re
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
inventory = json.loads((ROOT / 'reports/dpo_expanded/evidence/spreadsheet_paper_inventory.json').read_text())
originals = {
    'm826DekCpp': '4249_Property_Driven_Protein_I.pdf',
    'btEiAfnLsX': '24413_Why_DPO_is_a_Misspecifie.pdf',
    'cMEnMVvMw9': '6576_Token_Importance_Guided_D.pdf',
    'k79Un1LSXy': '362_Mitigating_Mismatch_within.pdf',
    'SD8Z231C45': '6741_DuPO_Enabling_Reliable_Se.pdf',
    'HEFPwoGtTj': '24535_Importance_Sampling_for_.pdf',
    'MoJSnVZ59d': '5041_SafeDPO_A_Simple_Approach.pdf',
    'G7DBGlgjjp': '11934_Uni_DPO_A_Unified_Paradi.pdf',
    'ekOwxTn65Y': '8851_P_2_DPO_Grounding_Halluci.pdf',
    'OuMNJoKJBQ': '15113_Alignment_Weighted_DPO_A.pdf',
    'tz9mJmgrdM': '24230_Is_On_Policy_Data_always.pdf',
}
records = []
for row in inventory['rows']:
    url = row['urls'][0]
    if 'arxiv.org/' in url:
        identifier = url.rsplit('/', 1)[1]
        matches = sorted((ROOT / 'resources').glob(identifier + 'v*.pdf'))
        assert len(matches) == 1, (identifier, matches)
        path = matches[0]
        status = 'downloaded_through_connected_browser'
    else:
        identifier = url.split('id=')[1]
        path = ROOT / 'resources' / originals[identifier]
        status = 'existing_pdf_reused'
    data = path.read_bytes()
    assert data.startswith(b'%PDF-'), path
    info = subprocess.check_output(['pdfinfo', str(path)], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
    assert pages > 0, path
    records.append({**row, 'identifier': identifier, 'file': str(path.relative_to(ROOT)),
                    'status': status, 'pages': pages, 'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest(),
                    'version_note': ('Spreadsheet links the ICLR 2025 submission although the tab is named ICLR 2026; existing local SafeDPO manuscript is preserved. See discussion audit for separate rounds.' if identifier == 'MoJSnVZ59d' else None)})
manifest = {'spreadsheet': inventory['source'], 'verified_on': datetime.now(ZoneInfo('Asia/Bangkok')).date().isoformat(),
            'listed_entries': len(records), 'distinct_paper_titles': len({r['title'] for r in records}),
            'downloaded_pdf_files': sum(r['status'].startswith('downloaded') for r in records),
            'existing_pdf_files_reused': sum(r['status'].startswith('existing') for r in records),
            'records': records}
(ROOT / 'resources/download_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
lines = ['# Spreadsheet paper resources', '',
         f"Verified on {manifest['verified_on']}: all 27 listed entries are present (26 distinct papers; AW-DPO is listed twice).", '',
         'Sixteen arXiv PDFs were downloaded automatically through the connected Chrome browser. Eleven existing PDFs were reused. The JSON manifest records source rows, versions, page counts, sizes, and SHA-256 hashes. This verifier checks local files; it does not require OpenReview API access.', '',
         'SafeDPO source discrepancy: the spreadsheet links the 2025 submission inside its ICLR 2026 tab. The local manuscript is preserved; the analysis distinguishes submission rounds.', '',
         '| Sheet / row | Paper | Local PDF |', '|---|---|---|']
for r in records:
    filename = Path(r['file']).name
    lines.append(f"| {r['sheet']} / {r['row']} | {r['title']} | [{filename}]({filename}) |")
lines += ['', 'Verification: `python3 reports/dpo_expanded/verify_corpus.py` from the workspace.', '',
          'Research outputs: [expanded report directory](../reports/dpo_expanded/).', '']
(ROOT / 'resources/README.md').write_text('\n'.join(lines))
print(json.dumps({k: manifest[k] for k in ('listed_entries', 'distinct_paper_titles', 'downloaded_pdf_files', 'existing_pdf_files_reused')}, indent=2))
