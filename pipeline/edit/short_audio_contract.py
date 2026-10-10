"""Source-bound directly conformed Short audio; no model calls or media writes."""
import json
import math


def gap_frames(beat):
    value = beat.get('gapAfterFrames', 0)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 48:
        raise ValueError('Short gaps must be integer frames between zero and48')
    return value


def gap_room_tone(spec, source_voice, frames):
    """Fill an authored pause from a hash-bound quiet window of the same voice."""
    import hashlib
    import numpy as np
    if not frames:
        return np.empty((0, 2), dtype=source_voice.dtype)
    bound = spec.get('gapRoomTone', {})
    start, count = bound.get('sourceStartSample'), bound.get('samples')
    if (any(isinstance(x, bool) or not isinstance(x, int) for x in (start, count))
            or start < 0 or not 3840 <= count <= 48000 or start + count > len(source_voice)):
        raise ValueError('A source-bound quiet room-tone window is required')
    tone = source_voice[start:start+count]
    digest = hashlib.sha256(tone.astype('<f4').tobytes()).hexdigest()
    if digest != bound.get('pcmSha256') or not np.isfinite(tone).all():
        raise ValueError('Changed room-tone source PCM')
    if float(np.sqrt(np.mean(tone.astype(np.float64)**2))) > .01:
        raise ValueError('Room tone is too loud for a narration gap')
    n = frames * 2000
    result = np.tile(tone, (math.ceil(n/count), 1))[:n].copy()
    # Only the copied quiet bed fades; the performed words remain untouched.
    ramp = min(480, n//2)
    result[:ramp] *= np.linspace(0, 1, ramp)[:, None]
    result[-ramp:] *= np.linspace(1, 0, ramp)[:, None]
    return result


def question_text(spec):
    value = spec.get('question')
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Omit question explicitly; do not render an empty raster')
    return value


def cta_filter(spec):
    mode = spec.get('ctaVisibility', 'legacy-fades')
    if mode == 'full-duration':
        return None
    if mode != 'legacy-fades' or spec['ctaFrames'] != 156:
        raise ValueError('Unknown CTA policy or unqualified legacy fade duration')
    return 'fade=t=in:d=0.25,fade=t=out:st=5.5:d=1'


def verify_whole_voice(spec, source_events, short_events, source_voice, voice):
    """Compare actual PCM against the already conformed whole source performances."""
    import numpy as np
    expected = np.zeros((spec['totalFrames'] * 2000, 2), dtype=voice.dtype)
    source_by_id = {e['id']: e for e in source_events}
    short_by_id = {e['id']: e for e in short_events}
    if len(source_by_id) != len(source_events) or len(short_by_id) != len(short_events):
        raise ValueError('Ambiguous source or Short event IDs')
    if set(short_by_id) != {b['id'] for b in spec['beats']}:
        raise ValueError('Short voice inventory differs from selected beats')
    if not source_events or not spec['beats'] or spec['beats'][0]['id'] != source_events[0]['id']:
        raise ValueError('Short must open with the film\'s first complete performance')
    source_positions = {e['id']: i for i, e in enumerate(source_events)}
    cursor = 0
    previous_source_position = -1
    rows = []
    for b in spec['beats']:
        e, se = source_by_id[b['id']], short_by_id[b['id']]
        position = source_positions[b['id']]
        if position <= previous_source_position:
            raise ValueError('Selected source performances must remain in chronological order')
        previous_source_position = position
        if b['startFrame'] != cursor or b['frames'] <= 0:
            raise ValueError('Short beats must be ordered and contiguous')
        if round(e['start'] * 24) != b['sourceStartFrame'] or round(se['start'] * 24) != cursor:
            raise ValueError('Changed source or destination voice clock')
        for key in ('audio', 'audioSha256', 'audioSourceStart', 'audioSourceDuration', 'text'):
            if e.get(key) != se.get(key):
                raise ValueError('Short changed a performed voice source: ' + key)
        duration = e['audioSourceDuration']
        if (e.get('audioSourceStart', 0) != 0 or not math.isfinite(duration)
                or duration > b['frames']/24 + 1/48000
                or abs(duration-b['audioDuration']) > 1/48000
                or e['text'] != b['spokenTextHi']):
            raise ValueError('Short must retain the exact complete spoken performance')
        src, dst, n = b['sourceStartFrame']*2000, cursor*2000, b['frames']*2000
        source_end = e['end']
        if position + 1 < len(source_events):
            source_end = min(source_end, source_events[position + 1]['start'])
        if (not math.isfinite(source_end)
                or src + n > round(source_end * 48000) + 1):
            raise ValueError('Selected source window crosses into an unselected performance')
        part = source_voice[src:src+n]
        if part.shape != (n, 2):
            raise ValueError('Source voice window is missing')
        expected[dst:dst+n] = part
        gap = gap_frames(b)
        if gap:
            expected[dst+n:dst+n+gap*2000] = gap_room_tone(spec, source_voice, gap)
        rows.append(dict(id=b['id'], sourceStartSample=src, outputStartSample=dst,
                         samples=n, voiceSamplesUnchanged=True, gapAfterFrames=gap))
        cursor += b['frames'] + gap
    if cursor != spec['bodyFrames'] or voice.shape != expected.shape:
        raise ValueError('Short voice length differs from body plus CTA')
    if not np.isfinite(voice).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite voice samples')
    error = float(np.max(np.abs(voice.astype(np.float64)-expected.astype(np.float64))))
    if error > 2e-7:
        raise ValueError('Short altered whole voice samples or put speech on the CTA')
    return rows, error


def prepared_audio(spec, inputs, decode):
    """All input files have already passed the renderer's path/hash checks."""
    import numpy as np
    read = lambda key: json.loads(inputs[key].read_text())
    audit, voice_audit, source_audit = read('mixAudit'), read('voiceAudit'), read('sourceVoiceAudit')
    if (audit.get('kind') not in ('working-audio-context-not-final', 'source-bound-score-sfx-revision')
            or audit.get('voiceGain') != 1
            or not math.isfinite(audit.get('unchangedVoiceRecoveryMaxError', float('nan')))
            or audit['unchangedVoiceRecoveryMaxError'] >= 2e-6):
        raise ValueError('Prepared Short mix lacks linear whole-voice provenance')
    bindings = spec['inputs']
    if (audit['inputs']['clock'] != bindings['clock']
            or audit['inputs']['voice'] != bindings['voice']
            or audit['outputs']['working-premaster-48kf32.wav']['sha256'] != bindings['premix']['sha256']
            or audit['outputs']['working-premaster-48kf32.wav']['path'] != bindings['premix']['path']
            or voice_audit['outputs']['voice']['sha256'] != bindings['voice']['sha256']
            or voice_audit['clockSha256'] != bindings['clock']['sha256']
            or source_audit['outputs']['voice']['sha256'] != bindings['sourceVoice']['sha256']
            or source_audit['clockSha256'] != bindings['sourceClock']['sha256']):
        raise ValueError('Prepared Short receipts do not bind these exact stems and clocks')
    if audit['samples'] != spec['totalFrames']*2000 or audit['sampleRate'] != 48000:
        raise ValueError('Prepared Short receipt has the wrong sample clock')
    voice, premix, source = (decode(inputs[k]) for k in ('voice', 'premix', 'sourceVoice'))
    expected = (spec['totalFrames']*2000, 2)
    if voice.shape != expected or premix.shape != expected or not np.isfinite(premix).all():
        raise ValueError('Prepared Short PCM must exactly cover body plus CTA')
    rows, error = verify_whole_voice(spec, read('sourceClock')['events'],
                                     read('clock')['events'], source, voice)
    return premix, voice, rows, error
