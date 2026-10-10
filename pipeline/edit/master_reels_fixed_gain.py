#!/usr/bin/env python3
"""Finite source-bound Reels-only mastering qualification on a qualified GPU worker.

No automatic loudnorm gain riding, picture inference, threshold relaxation or
claim of subjective hearing. Each candidate starts from the same lossless mix.
"""
import argparse
from array import array
import json
import math
from pathlib import Path
import platform
import socket
import subprocess

import numpy as np
import audit_master_voice_sync as sync
import gpu_finishing_probe as gpu
import mix_absolute_sfx_longform as mix


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, capture_output=True, **kwargs)


def limiter(limit, release):
    base = f'alimiter=limit={limit}:attack=5:release={release}:level=false'
    signal = array('f', [0.0]) * 48000
    signal[10000] = .1
    result = run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', '192000',
                  '-ac', '1', '-i', 'pipe:0', '-af', base, '-f', 'f32le', 'pipe:1'],
                 input=signal.tobytes())
    decoded = array('f'); decoded.frombytes(result.stdout)
    delay = max(range(len(decoded)), key=lambda i: abs(decoded[i])) - 10000
    assert len(decoded) == len(signal) and 0 <= delay <= 3840
    compensation = f',atrim=start_sample={delay},asetpts=N/SR/TB,apad=pad_len={delay}'
    return base + compensation, dict(sampleRate=192000, measuredDelaySamples=delay,
                                    compensation='Leading measured delay removed; same samples padded at tail')


def voice_rows(candidate, reference, plan):
    rows = []
    for event in plan['events']:
        lo = event['start'] + event['subtitleAudioAt'] + .2
        hi = event['start'] + event['subtitleAudioAt'] + event['subtitleAudioDuration'] - .2
        start, end = round(lo * sync.SR), round(hi * sync.SR)
        begin, stop = max(0, start-1200), min(len(candidate), end+1200)
        position, corr = sync.compare(reference[start:end], candidate[begin:stop])
        lag = (begin + position - start) / sync.SR
        rows.append(dict(event=event['id'], windowSeconds=[lo, hi],
                         lagSeconds=lag, correlation=corr,
                         passed=abs(lag) <= .001 and corr >= .85))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    assert platform.system() == 'Linux', 'All media processing belongs on the qualified GPU worker'
    gpu_name = run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).stdout.strip()
    assert gpu_name
    ep = args.episode.resolve(); config_path = (ep / args.config).resolve()
    config = json.loads(config_path.read_text())
    sync.SR = 48000
    assert config['kind'] == 'finite-fixed-gain-reels-qualification'
    assert config['generationApiAllowed'] is False
    bindings = {str(p): sync.sha(p) for p in (Path(__file__), config_path,
                Path(sync.__file__), Path(gpu.__file__), Path(mix.__file__))}
    paths = {}
    for key, spec in config['inputs'].items():
        path = (ep / spec['path']).resolve(); path.relative_to(ep)
        assert sync.sha(path) == spec['sha256'], key
        paths[key] = path; bindings[str(path)] = spec['sha256']
    out = (ep / config['outputDirectory']).resolve(); out.relative_to(ep / 'work')
    out.mkdir(parents=True, exist_ok=False)
    assert 1 <= len(config['candidates']) <= 8
    plan = json.loads(paths['voicePlan'].read_text())
    frames = config['expectedVideoFrames']; assert type(frames) is int and frames > 0
    duration = frames / 24; samples = frames * 2000
    assert abs(plan['runtimeSeconds']-duration) < 1e-8
    assert [x['id'] for x in plan['events']] == config['expectedEventIds']
    source_probe = gpu.probe_media(paths['picture'])
    assert len(source_probe['video']) == 1 and not source_probe['audio']
    assert source_probe['video'][0]['frames'] == frames
    premaster = mix.decode_samples(paths['premaster']); voice48 = mix.decode_samples(paths['voice'])
    assert len(premaster) == len(voice48) == samples
    assert np.isfinite(premaster).all() and np.isfinite(voice48).all()
    cta_seconds = float(config['ctaSeconds'])
    assert math.isfinite(cta_seconds) and 0 < cta_seconds <= duration
    cta_start = samples - round(cta_seconds*48000)
    assert np.max(np.abs(voice48[cta_start:])) == 0, 'CTA must contain no narration'
    voice = sync.load(paths['voice'], True)
    controlled = out / 'controlled-48kf32.wav'
    ratio=config.get('compressorRatio',2)
    assert type(ratio) in (int,float) and 1<=ratio<=4
    compressor = f'acompressor=threshold=0.10:ratio={ratio}:attack=5:release=120:makeup=1'
    run(['ffmpeg','-v','error','-i',paths['premaster'],'-af',compressor,
         '-ar','48000','-ac','2','-c:a','pcm_f32le',controlled])
    assert len(mix.decode_samples(controlled)) == samples
    settings = dict(targetIntegratedLufs=-10,targetTruePeakDbtp=-2.5,targetLra=11)
    results = []
    for spec in config['candidates']:
        assert type(spec['gainDb']) in (int,float) and -6 <= spec['gainDb'] <= 24
        assert .5 <= spec['limit'] <= .75
        assert spec['releaseMs'] in (10,20) and 0 <= spec['postGainDb'] <= 2.5
        limiter_filter, proof = limiter(spec['limit'], spec['releaseMs'])
        proof['releaseMs'] = spec['releaseMs']
        filters = (f"volume={spec['gainDb']}dB,aresample=192000," + limiter_filter +
                   f",aformat=channel_layouts=stereo,aresample=48000,volume={spec['postGainDb']}dB,asetpts=N/SR/TB")
        lossless = out / (spec['id'] + '.wav'); encoded = out / (spec['id'] + '.m4a')
        run(['ffmpeg','-v','error','-i',controlled,'-af',filters,'-ar','48000',
             '-ac','2','-c:a','pcm_f32le',lossless])
        pcm = mix.decode_samples(lossless); assert len(pcm) == samples and np.isfinite(pcm).all()
        run(['ffmpeg','-v','error','-i',lossless,'-c:a','aac','-b:a','384k',
             '-ar','48000','-ac','2','-movflags','+faststart',encoded])
        measured = mix.loudness(encoded,settings)
        rows = voice_rows(sync.load(encoded, True),voice,plan)
        integrated, peak = float(measured['input_i']),float(measured['input_tp'])
        result = dict(**spec,filters=filters,limiterQualification=proof,
                      losslessPath=str(lossless.relative_to(ep)),losslessSha256=sync.sha(lossless),
                      encodedPath=str(encoded.relative_to(ep)),encodedSha256=sync.sha(encoded),
                      loudness=measured,rawVoiceSpeechBand=rows,
                      minimumRawVoiceCorrelation=min(x['correlation'] for x in rows),
                      maxAbsoluteLagSeconds=max(abs(x['lagSeconds']) for x in rows),
                      passed=abs(integrated+10)<=.5 and peak<=-1 and all(x['passed'] for x in rows))
        results.append(result)
        (out / (spec['id'] + '.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('id','loudness','minimumRawVoiceCorrelation','maxAbsoluteLagSeconds','passed')}),flush=True)
    passed = [r for r in results if r['passed']]
    report = dict(kind='finite-fixed-gain-reels-qualification',computeHost=socket.gethostname(),gpu=gpu_name,
                  bindings=bindings,frames=frames,fps=24,runtimeSeconds=duration,
                  controlledSha256=sync.sha(controlled),compressor=compressor,
                  candidates=results,passed=bool(passed),
                  thresholds=dict(integratedLufs=[-10.5,-9.5],maximumTruePeakDbtp=-1,
                                  minimumOriginalVoiceCorrelation=.85,maxLagSeconds=.001),
                  scope='Reels only. No YouTube modification. No subjective listening or owner viewing claimed.')
    if not passed:
        (out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
        raise RuntimeError('No finite candidate met unchanged Reels and original-voice requirements')
    # Prefer fidelity, not the loudest passing result or an outcome-conditioned threshold.
    chosen = max(passed,key=lambda r:r['minimumRawVoiceCorrelation'])
    mastered = out/'short-reels.mp4'; audio = ep/chosen['encodedPath']
    run(['ffmpeg','-v','error','-i',paths['picture'],'-i',audio,'-map','0:v:0','-map','1:a:0',
         '-c','copy','-t',f'{duration:.9f}','-movflags','+faststart',mastered])
    picture_hash = mix.packet_hash(paths['picture'])
    assert mix.packet_hash(mastered) == picture_hash
    final_probe = gpu.probe_media(mastered)
    assert final_probe['video'][0]['frames'] == frames
    assert final_probe['audio'][0]['sampleRate'] == 48000 and final_probe['audio'][0]['channels'] == 2
    assert abs(final_probe['durationSeconds']-duration) < .05
    final_loudness = mix.loudness(mastered,settings)
    assert abs(float(final_loudness['input_i'])+10)<=.5 and float(final_loudness['input_tp'])<=-1
    final_rows = voice_rows(sync.load(mastered,True),voice,plan)
    assert all(x['passed'] for x in final_rows)
    # Independently compare encoded master against its exact lossless transform.
    expected = sync.load(ep/chosen['losslessPath']); actual = sync.load(mastered)
    fidelity_rows = voice_rows(actual,expected,plan)
    assert all(abs(x['lagSeconds']) <= .001 and x['correlation'] >= .99 for x in fidelity_rows)
    decode = gpu.decode_h264_worker(mastered,frames)
    assert bindings == {p:sync.sha(Path(p)) for p in bindings}
    report.update(chosenCandidate=chosen['id'],output=str(mastered.relative_to(ep)),
                  outputSha256=sync.sha(mastered),probe=final_probe,loudness=final_loudness,
                  picturePacketsUnchanged=True,picturePacketSha256=picture_hash,fullDecode=decode,
                  rawOriginalVoiceSpeechBand=final_rows,encodedVsLosslessTransform=fidelity_rows,
                  originalVoiceUnmodified=True,ctaVoiceExactZero=True,
                  finalApproved=False,creativeApproval=False,continuousListeningPerformed=False)
    (out/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=report['output'],sha256=report['outputSha256'],chosen=chosen['id'],
                          loudness=final_loudness,passed=True)),flush=True)


if __name__ == '__main__':
    main()

