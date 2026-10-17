#!/usr/bin/env python3
"""Carry measured speech QA through proven AAC-preserving picture changes.

This is explicitly derived evidence, NOT a new voice correlation measurement.
The reference final/body receipts must be retained alongside this result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def packet_times(path, stream):
    p = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', stream, '-show_packets',
        '-show_entries', 'packet=pts_time', '-of', 'json', str(path)], text=True))
    return sorted(float(x['pts_time']) for x in p['packets'])


def uniform_shift(before, after, tolerance=.002):
    require(len(after) >= len(before) > 0, 'Missing source packets')
    shifts = [b-a for a,b in zip(before, after)]
    require(max(shifts)-min(shifts) <= tolerance, 'Non-uniform presentation-clock change')
    return {'min': min(shifts), 'max': max(shifts), 'packets': len(before)}


def audio_hash(path, packets):
    return subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
        '-frames:a', str(packets), '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-'
    ], text=True).strip()


def derive_rows(reference, audio_delta):
    rows = []
    for row in reference['results']:
        require(row['pass'] is True, 'Reference speech event failed')
        lo = row['lagSeconds'] + audio_delta['min']
        hi = row['lagSeconds'] + audio_delta['max']
        bound = max(abs(lo), abs(hi))
        rows.append({**row, 'lagSeconds': (lo+hi)/2,
                     'lagBoundsSeconds': [lo,hi], 'maxAbsoluteLagBoundSeconds': bound,
                     'correlationIsInherited': True,
                     'pass': bound <= reference['maxAllowedLagSeconds'] and
                             row['correlation'] >= reference['minimumAllowedCorrelation']})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source_body', 'master', 'plan', 'reference_sync',
                 'reference_append', 'reference_append_clock', 'output'):
        ap.add_argument(name, type=Path)
    a = ap.parse_args()
    require(platform.system() == 'Linux', 'Media QA belongs on GCP, not the Mac')
    subprocess.run(['nvidia-smi', '-L'], capture_output=True, check=True)
    require(not a.output.exists(), 'Use a new immutable audit path')
    documents = {name: json.loads(getattr(a,name).read_text()) for name in
                 ('plan','reference_sync','reference_append','reference_append_clock')}
    plan, ref, append, clock = (documents[x] for x in
                               ('plan','reference_sync','reference_append','reference_append_clock'))
    bindings = {str(getattr(a,n)): sha(getattr(a,n)) for n in
                ('source_body','master','plan','reference_sync','reference_append','reference_append_clock')}
    source_sha, final_sha = bindings[str(a.source_body)], bindings[str(a.master)]
    require(ref['passed'] and ref['timingPassed'] and ref['voicePresencePassed'], 'Reference QA failed')
    require(ref['planSha256'] == bindings[str(a.plan)], 'Reference plan changed')
    require(append['coreSha256'] == source_sha == clock['bodySha256'], 'Reference body identity mismatch')
    require(append['sha256'] == ref['masterSha256'] == clock['finalSha256'], 'Reference final identity mismatch')
    require(append['bodyAudioPacketsUnchanged'] and append['bodyPicturePacketsUnchanged'] and
            append['decode'] == 'passed' and clock['passed'], 'Reference append was not verified')
    voiced = [e for e in plan['events'] if e.get('kind') != 'picture']  # wordless picture has no voice
    require(ref['events'] == len(voiced) == len(ref['results']), 'Incomplete reference scene coverage')
    require([r['event'] for r in ref['results']] == [e['id'] for e in voiced], 'Scene order changed')
    shifts = {}
    for stream in ('v:0','a:0'):
        shifts[stream] = uniform_shift(packet_times(a.source_body,stream), packet_times(a.master,stream))
        require(shifts[stream]['packets'] == clock['streams'][stream]['bodyPackets'], 'Reference packet count changed')
    packets = shifts['a:0']['packets']
    source_audio = audio_hash(a.source_body, packets)
    require(source_audio == audio_hash(a.master, packets), 'AAC payload changed: inherited correlation is invalid')
    old_a, old_v = clock['streams']['a:0'], clock['streams']['v:0']
    new_a, new_v = shifts['a:0'], shifts['v:0']
    delta = {'min': new_a['min']-old_a['maxShiftSeconds'],
             'max': new_a['max']-old_a['minShiftSeconds']}
    video_delta = max(abs(new_v[x]-old_v[y]) for x in ('min','max')
                      for y in ('minShiftSeconds','maxShiftSeconds'))
    require(video_delta <= .002, 'New final presentation clock differs from reference')
    rows = derive_rows(ref,delta)
    require(all(r['pass'] for r in rows), 'Derived speech alignment exceeds reference bounds')
    require(all(sha(p)==h for p,h in bindings.items()), 'Inputs changed during verification')
    result = {**{k:v for k,v in ref.items() if k not in ('master','masterSha256','method','results')},
              'master': str(a.master.resolve()), 'masterSha256': final_sha,
              'method': 'Inherited measured speech QA through exact AAC body-payload identity and every-packet presentation-clock preservation; NOT remeasured voice correlation.',
              'inherited': True, 'referenceMeasuredMasterSha256': ref['masterSha256'],
              'referenceVoiceAvailableOnThisHost': False,
              'bindings': bindings, 'sourceBodyAudioPacketHash': source_audio,
              'presentationShifts': shifts, 'audioShiftDeltaSeconds': delta,
              'maxVideoClockChangeFromReferenceSeconds': video_delta,
              'maxAbsoluteLagSeconds': max(r['maxAbsoluteLagBoundSeconds'] for r in rows),
              'minimumCorrelation': min(r['correlation'] for r in rows),
              'timingPassed': True, 'voicePresencePassed': True, 'passed': True,
              'results': rows,
              'limits': 'No new listening or vocal-quality claim. Correlations are inherited from the identified actual reference measurement, not calculated against a private voice stem on this host.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))


if __name__ == '__main__':
    main()
