#!/usr/bin/env python3
"""Build one recording-level comparison CSV per ASR run (standard library only).
Use final, QC-filtered human scores and ASR scores with matching normalization
and name-exclusion settings. This script joins existing scores; it never rescores.
"""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean

FIELDS = ['audio_file', 'utterance_id', 'condition', 'speaker_id', 'gender',
          'language', 'asr_system', 'human_n_ratings', 'human_mean_wer',
          'human_mean_cer', 'asr_wer', 'asr_cer']


def read_csv(path, required):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: no rows')
    return rows


def path_key(value):
    if not value or not value.strip():
        raise ValueError('Blank audio_file')
    return str(PurePosixPath(value.strip().replace('\\', '/')))


def rate(value, context, allow_blank=False):
    if value is None or not value.strip():
        if allow_blank:
            return ''
        raise ValueError(f'{context}: missing score')
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{context}: invalid error rate {value!r}')
    return result  # WER/CER may legitimately exceed 1.


def language_key(value):
    return {'en': 'eng', 'english': 'eng'}.get(value.strip().lower(), value.strip().lower())


def prepare(args):
    required = ['participant_id', 'session_id', 'audio_file', 'utterance_id',
                'condition', 'speaker_id', 'gender', 'language', 'wer', 'cer',
                'reference_text_normalized']
    groups = defaultdict(list)
    seen = set()
    skipped = 0
    for row in read_csv(args.human, required):
        flag = row.get('is_attention_check', '').strip().lower()
        if flag in {'true', '1', '1.0', 'yes'}:
            skipped += 1
            continue
        if flag not in {'false', '0', '0.0', 'no', ''}:
            raise ValueError(f'Unrecognized attention-check flag: {flag!r}')
        key = path_key(row['audio_file'])
        if not row['participant_id'].strip():
            raise ValueError(f'Blank participant_id for {key}')
        response = (row['participant_id'], key)
        if response in seen:
            raise ValueError(f'Duplicate listener/recording: {response}; resolve repeated sessions first')
        seen.add(response)
        groups[key].append(row)
    human = {}
    for key, rows in groups.items():
        for field in ['utterance_id', 'condition', 'speaker_id', 'gender', 'language',
                      'reference_text_normalized']:
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f'Conflicting human {field} for {key}')
        first = rows[0]
        human[key] = {f: first[f] for f in ['audio_file', 'utterance_id', 'condition',
                                           'speaker_id', 'gender', 'language']}
        human[key].update(human_n_ratings=len(rows),
            human_mean_wer=mean(rate(r['wer'], key) for r in rows),
            human_mean_cer=mean(rate(r['cer'], key) for r in rows))
    if not human:
        raise ValueError('No eligible human rows')

    mapping = {}
    for row in read_csv(args.mapping, ['model_name', 'condition', 'utt_id', 'audio_file']):
        k = (row['model_name'], row['utt_id'])
        if k in mapping:
            raise ValueError(f'Duplicate model/utterance in mapping: {k}')
        mapping[k] = row

    tables, report, labels = [], [], set()
    report.append(f'Human responses: {len(seen)}; recordings: {len(human)}; attention rows skipped: {skipped}')
    for spec in args.asr:
        if '=' not in spec:
            raise ValueError('--asr must be LABEL=DIRECTORY')
        label, location = spec.split('=', 1)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', label) or label in labels:
            raise ValueError(f'Invalid or duplicate ASR label: {label!r}')
        labels.add(label)
        root = Path(location)
        files = [root] if root.is_file() else sorted(root.rglob('asr.csv'))
        if not files:
            raise ValueError(f'No asr.csv files under {root}')
        objective, seen_asr = {}, set()
        extra = 0
        for file in files:
            for row in read_csv(file, ['utt_id', 'model_name', 'speaker_id', 'language',
                                       'wer', 'cer', 'ref_text_normalized']):
                k = (row['model_name'], row['utt_id'])
                if k in seen_asr:
                    raise ValueError(f'{label}: duplicate ASR model/utterance {k}')
                seen_asr.add(k)
                if k not in mapping:
                    raise ValueError(f'{label}: no mapping for {k}')
                link = mapping[k]
                key = path_key(link['audio_file'])
                if key not in human:
                    extra += 1
                    continue
                h = human[key]
                if h['utterance_id'] != row['utt_id'] or h['condition'] != link['condition']:
                    raise ValueError(f'{label}: utterance/condition mismatch for {key}')
                if h['speaker_id'] != row['speaker_id'] or language_key(h['language']) != language_key(row['language']):
                    raise ValueError(f'{label}: speaker/language mismatch for {key}')
                if groups[key][0]['reference_text_normalized'] != row['ref_text_normalized']:
                    raise ValueError(f'{label}: normalized references differ for {key}; inspect scoring inputs')
                if key in objective:
                    raise ValueError(f'{label}: multiple ASR results map to {key}')
                objective[key] = (rate(row['wer'], str(k), True), rate(row['cer'], str(k), True))
        output = []
        complete = 0
        for key, h in sorted(human.items()):
            wer, cer = objective.get(key, ('', ''))
            complete += wer != '' and cer != ''
            output.append({**h, 'asr_system': label, 'asr_wer': wer, 'asr_cer': cer})
            if wer == '' or cer == '':
                report.append(f'{label}: missing result or score: {h["condition"]} / {h["utterance_id"]}')
        report.append(f'{label}: {len(files)} input files; {complete}/{len(human)} complete matches; {extra} ASR-only recordings omitted')
        tables.append((f'comparison_{label}.csv', output))
    # Validate everything before writing. Require a fresh directory.
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError(f'Output must be a new or empty directory: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in tables:
        with (args.output / filename).open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    (args.output / 'join_summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--human', required=True, type=Path)
    p.add_argument('--mapping', required=True, type=Path)
    p.add_argument('--asr', required=True, action='append', metavar='LABEL=DIRECTORY')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    try:
        prepare(args)
    except (ValueError, OSError) as error:
        p.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
