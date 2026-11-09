"""Build the two explanatory notebooks, embed charts, and validate references."""
import base64
import json
import pathlib
import re

import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    report = json.loads((ROOT/'artifacts/data_audit.json').read_text(encoding='utf-8'))
    # Narrative counts must be reviewed, not silently kept when analysis changes.
    expected = {'candidate_records':1098209,'valid_numeric_records':1098208,
                'repaired_trailing_commas':11733,'exact_duplicate_occurrences':66820,
                'zero_timestamps':12845}
    for key, value in expected.items():
        if report[key] != value:
            raise ValueError(f'Review narrative for changed audit statistic: {key}')
    assert len(report['invalid_records']) == 1
    assert len(report['subject_class_counts']) == 36
    assert len(report['missing_classes_by_subject']) == 17
    assert report['class_counts'] == dict(Walking=424399,Jogging=342179,Upstairs=122869,
                                         Downstairs=100427,Sitting=59939,Standing=48395)
    drafts = json.loads((ROOT/'scripts/notebook_content.json').read_text(encoding='utf-8'))
    planned = set(drafts)
    for filename, entries in drafts.items():
        cells=[]
        for index, entry in enumerate(entries):
            source=entry['source']
            attachments={}
            for match in list(re.finditer(r'!\[([^\]]*)\]\((artifacts/[^)]+\.png)\)',source)):
                alt, relative=match.groups()
                path=ROOT/relative
                attachments[path.name]={'image/png':base64.b64encode(path.read_bytes()).decode('ascii')}
                source=source.replace(match.group(0),f'![{alt}](attachment:{path.name})')
            for target in re.findall(r'\]\(([^)]+)\)',source):
                if target.startswith(('https://','http://','attachment:','#')):
                    continue
                if target not in planned and not (ROOT/target).exists():
                    raise ValueError(f'Broken local link: {filename}: {target}')
            cell=nbformat.v4.new_markdown_cell(source)
            cell['id']=f'cell-{index:02d}'
            if attachments:
                cell['attachments']=attachments
            cells.append(cell)
        notebook=nbformat.v4.new_notebook(cells=cells,metadata={
            'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python'},
            'wisdm':{'status':'audit_and_design_only','date':'2026-11-02',
                     'archive_sha256':report['archive_sha256']}})
        nbformat.validate(notebook)
        nbformat.write(notebook,ROOT/filename)
        reread=nbformat.read(ROOT/filename,as_version=4)
        nbformat.validate(reread)
        assert not any(c.cell_type=='code' for c in reread.cells)
        print(f'{filename}: {len(cells)} markdown cells; format and links validated')

if __name__=='__main__':
    main()
