"""Audit WISDM without changing/extracting raw data. Run from any directory."""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
LABELS = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
OFFICIAL = [424400, 342177, 122869, 100427, 59939, 48395]

def analyze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source) as archive:
        raw = archive.extractfile('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt').read()
        docs = {pathlib.PurePosixPath(m.name).name: archive.extractfile(m).read().decode()
                for m in archive.getmembers() if m.isfile() and ('about' in m.name or 'readme' in m.name)}
    counts = np.zeros((36, 6), dtype=np.int64)
    candidates = repaired = zeros = duplicates = 0
    invalid, deltas, xyz, samples = [], [], [], {}
    seen = set()
    previous = None
    for token in re.split(r'[;\r\n]+', raw.decode()):
        token = token.strip()
        if not token:
            continue
        candidates += 1
        fields = token.split(',')
        if len(fields) == 7 and fields[-1].strip() == '':
            repaired += 1
            fields.pop()
        try:
            if len(fields) != 6:
                raise ValueError('expected six fields')
            user, label, ts, *acc = fields
            user, ts = int(user), int(ts)
            acc = tuple(float(v) for v in acc)
            if not 1 <= user <= 36 or label not in LABELS or not np.isfinite(acc).all():
                raise ValueError('invalid user, label or acceleration')
        except ValueError as error:
            invalid.append(dict(record=candidates, text=token, reason=str(error)))
            previous = None
            continue
        record = (user, label, ts, *acc)
        duplicates += record in seen
        seen.add(record)
        counts[user-1, LABELS.index(label)] += 1
        zeros += ts == 0
        xyz.append(acc)
        if previous and previous[:2] == (user, label):
            deltas.append(ts - previous[2])
        previous = (user, label, ts)
        # Illustrative original-order snippets, not representative class prototypes.
        if label not in samples:
            samples[label] = {'user': user, 'xyz': []}
        if user == samples[label]['user'] and len(samples[label]['xyz']) < 128:
            samples[label]['xyz'].append(acc)
    dt = np.asarray(deltas, dtype=np.int64)
    positive = dt[dt > 0] / 1e6
    a = np.asarray(xyz)
    totals = counts.sum(axis=0)
    result = dict(source=str(source), archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  raw_sha256=hashlib.sha256(raw).hexdigest(),
                  parser='Split on semicolon/CR/LF; remove empty seventh field only; validate user, activity and finite xyz. No timestamp filtering or deduplication.',
                  candidate_records=candidates, valid_numeric_records=int(totals.sum()),
                  repaired_trailing_commas=repaired, invalid_records=invalid,
                  exact_duplicate_occurrences=duplicates, zero_timestamps=zeros,
                  class_counts=dict(zip(LABELS, map(int, totals))),
                  official_class_counts=dict(zip(LABELS, OFFICIAL)),
                  subject_class_counts={str(i+1):dict(zip(LABELS,map(int,row))) for i,row in enumerate(counts)},
                  missing_classes_by_subject={str(i+1):[LABELS[j] for j in range(6) if row[j]==0]
                                              for i,row in enumerate(counts) if (row==0).any()},
                  adjacent_same_subject_activity=dict(total=len(dt), zero=int((dt==0).sum()),
                      negative=int((dt<0).sum()), above_250ms=int((dt>250000000).sum()),
                      positive_ms_percentiles=dict(zip(['p1','p25','p50','p75','p99'],np.percentile(positive,[1,25,50,75,99]).tolist()))),
                  acceleration_summary={axis:dict(zip(['min','p1','median','p99','max'],np.percentile(a[:,i],[0,1,50,99,100]).tolist())) for i,axis in enumerate('xyz')},
                  status='Data audit only; no training, split assignment, resampling or final cleaning.')
    assert int(totals.sum()) + len(invalid) == candidates
    assert counts.shape == (36,6) and (counts.sum(axis=1)>0).all()
    (output/'data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'source_descriptions.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10, 'figure.dpi':140})
    fig, ax = plt.subplots(figsize=(9,4))
    ax.bar(LABELS,totals,color='#337ea9')
    for i,n in enumerate(totals):
        ax.text(i,n+6000,f'{n:,}\n{n/totals.sum():.1%}',ha='center',fontsize=9)
    ax.set(ylim=(0,510000),ylabel='Parsed sensor records',title='WISDM v1.1: activity distribution (before timestamp cleaning)')
    fig.tight_layout(); fig.savefig(output/'class_distribution.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,10))
    im=ax.imshow(counts,aspect='auto',cmap='Blues')
    ax.set(xticks=range(6),xticklabels=LABELS,yticks=range(36),yticklabels=range(1,37),ylabel='Subject ID',title='Subject × activity: parsed record counts')
    for i in range(36):
        for j in range(6):
            if counts[i,j]==0: ax.text(j,i,'0',ha='center',va='center',color='#b21d33',fontsize=8)
    fig.colorbar(im,ax=ax,label='Records'); fig.tight_layout(); fig.savefig(output/'subject_coverage.png'); plt.close(fig)
    fig, ax=plt.subplots(figsize=(9,4))
    ax.hist(positive[positive<=250],bins=100,color='#337ea9')
    ax.axvline(50,color='#bf4d2b',label='Nominal 50 ms')
    ax.set(xlabel='Positive adjacent interval (ms)',ylabel='Pairs',title='Same subject/activity, original order; only 0 < interval ≤ 250 ms shown')
    ax.legend(); fig.tight_layout(); fig.savefig(output/'time_intervals.png'); plt.close(fig)
    print(json.dumps({k:result[k] for k in ['candidate_records','valid_numeric_records','exact_duplicate_occurrences','acceleration_summary']},indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=pathlib.Path,default=pathlib.Path(r'C:\Users\ektlaksskgodqhrgo\Downloads\WISDM_ar_latest.tar.gz'))
    parser.add_argument('--output',type=pathlib.Path,default=ROOT/'artifacts')
    args=parser.parse_args()
    analyze(args.archive,args.output)
