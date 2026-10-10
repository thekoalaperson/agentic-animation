"""Build independent, non-final audio stems from a measured speech clock.

This does not turn an incomplete picture clock into a renderable edit plan.
The shared voice/score mechanics remain in assemble_longform.
"""
import argparse
import json
from pathlib import Path
import shutil
import socket
import sys
import math

import assemble_longform as assembly


def validate_working_clock(clock, episode):
    if clock.get('kind')!='measured-speech-clock-not-renderable-edit-plan' or clock.get('renderable') is not False or clock.get('episode')!=episode:
        raise RuntimeError('Working mix requires the exact non-final episode clock')
    events=clock.get('events',[]); runtime=float(clock['runtimeSeconds'])
    if not events or len(events)!=clock.get('eventCount') or not math.isfinite(runtime) or runtime<=0:
        raise RuntimeError('Incomplete measured event inventory')
    if len({e['id'] for e in events})!=len(events) or len({e['shot'] for e in events})!=len(events):
        raise RuntimeError('Ambiguous measured event identity')
    cursor=0.0
    for event in events:
        start,end,duration=[float(event[k]) for k in ('start','end','duration')]
        if not all(math.isfinite(v) for v in (start,end,duration)) or abs(start-cursor)>1/48000 or duration<=0 or abs(start+duration-end)>1/48000:
            raise RuntimeError('Working speech clock is not complete and contiguous')
        cursor=end
    if abs(cursor-runtime)>1/48000:
        raise RuntimeError('Working speech clock does not end at its declared runtime')


def mix_working(ep, config_path, config):
    """Use shared mixer DSP without promoting a working clock into picture."""
    import mix_absolute_sfx_longform as mix
    if config.get('finalApproved') is not False:
        raise RuntimeError('Working audio mix cannot claim final approval')
    original_config_sha=assembly.sha(config_path)
    paths={}
    for key,spec in config['inputs'].items():
        paths[key]=assembly.bound_path(ep,spec['path'],key)
        assembly.verify_sha(paths[key],spec['sha256'],key)
    clock=json.loads(paths['clock'].read_text())
    validate_working_clock(clock,config['episode'])
    stem_audit=json.loads(paths['stemAudit'].read_text()); sfx_audit=json.loads(paths['sfxAudit'].read_text())
    if stem_audit.get('finalApproved') is not False or sfx_audit.get('workingAudioOnly') is not True or sfx_audit.get('finalApproved') is not False:
        raise RuntimeError('Only explicitly working stems accepted')
    for key in ('voice','music'):
        if stem_audit['outputs'][key]['sha256']!=config['inputs'][key]['sha256']:
            raise RuntimeError('Working audio stem receipt mismatch')
    if sfx_audit['sha256']!=config['inputs']['sfx']['sha256'] or sfx_audit['clockSha256']!=config['inputs']['clock']['sha256']:
        raise RuntimeError('Working effects do not bind the current clock')
    runtime=float(clock['runtimeSeconds']); expected=round(runtime*mix.SR)
    for key in ('voice','music','sfx'):
        mix.verify_audio_probe(mix.probe_media(paths[key]),runtime,expected,key,codec='pcm_f32le' if key=='sfx' else 'pcm_s24le')
    output=mix.work_output(ep,config['outputDirectory'],'working audio mix directory')
    if output.name in ('assembly-v1','final-delivery-v1'):
        raise RuntimeError('Working mix cannot overwrite canonical assembly')
    output.mkdir(parents=True,exist_ok=True)
    settings=config['settings']
    if not -6<=float(settings['musicGainDb'])<=6:
        raise RuntimeError('Unbounded working score gain')
    for key in ('musicDuck','sfxDuck'):
        d=settings[key]
        if not all(math.isfinite(float(v)) for v in d.values()) or not .005<=d['threshold']<=.1 or not 1<=d['ratio']<=12 or not 1<=d['attackMs']<=100 or not 100<=d['releaseMs']<=1500:
            raise RuntimeError('Unbounded working sidechain')
    notches=mix.validate_notches(config.get('scoreNotchAutomation',[]),runtime)
    if notches:
        if 'speechProtectionAudit' not in paths:
            raise RuntimeError('Working score notches require source-bound screening evidence')
        previous=json.loads(paths['speechProtectionAudit'].read_text())
        prior_config=json.loads(paths['speechProtectionConfig'].read_text())
        assembly.verify_sha(paths['speechProtectionConfig'],previous['configSha256'],'screening config')
        if previous.get('kind')!='working-audio-context-not-final' or previous.get('scoreNotchAutomation'):
            raise RuntimeError('Protection must be a single pass from the original working mix')
        for key in ('clock','voice','music','sfx'):
            if prior_config['inputs'][key]!=config['inputs'][key]:
                raise RuntimeError('Speech protection evidence refers to different inputs')
    contract=dict(settings=settings,scoreNotchEnvelope={'ffmpegVolumeExpression':mix.notch_volume_expression(notches)})
    premaster=output/'working-premaster-48kf32.wav'
    temporary=premaster.with_suffix('.partial.wav')
    graph=mix.filter_graph(contract)
    mix.run(['ffmpeg','-y','-v','error','-i',str(paths['voice']),'-i',str(paths['music']),'-i',str(paths['sfx']),
             '-filter_complex',graph,'-map','[out]','-t',f'{runtime:.9f}','-ar',str(mix.SR),'-c:a','pcm_f32le',str(temporary)])
    temporary.replace(premaster)
    mix.verify_audio_probe(mix.probe_media(premaster),runtime,expected,'working premaster',codec='pcm_f32le')
    audit={key:mix.loudness(paths[key],settings) for key in ('voice','music','sfx')}
    audit['premaster']=mix.loudness(premaster,settings)
    audit['intelligibility']=mix.intelligibility(paths['voice'],premaster,clock['events'])
    for key,path in paths.items():
        assembly.verify_sha(path,config['inputs'][key]['sha256'],key)
    assembly.verify_sha(config_path,original_config_sha,'working mix config')
    result=dict(kind='working-audio-context-not-final',episode=config['episode'],finalApproved=False,
                renderablePictureCreated=False,computeHost=socket.gethostname(),configSha256=original_config_sha,
                sourceSha256=assembly.sha(Path(__file__)),mixerSha256=assembly.sha(Path(mix.__file__)),
                runtimeSeconds=runtime,settings=settings,filterGraph=graph,buses=audit,scoreNotchAutomation=notches,
                output=dict(path=str(premaster),sha256=assembly.sha(premaster)),
                scope='Unnormalized floating-point context mix and measured voice/background competition; no listening or final creative acceptance claim.')
    mix.atomic_json(output/'working-mix-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='buses'},indent=2))
    print(json.dumps(audit['intelligibility']['summary'],indent=2))


def main():
    if sys.platform != "linux" or not shutil.which("nvidia-smi"):
        raise RuntimeError("Production audio work belongs on the qualified GPU worker")
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    ep = args.episode.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get('kind') == 'working-audio-mix-only':
        return mix_working(ep, config_path, config)
    original_config_sha=assembly.sha(config_path)
    voice_only = config.get("kind") == "working-voice-only"
    if config.get("kind") not in ("working-audio-only", "working-voice-only") or config.get("finalApproved") is not False:
        raise RuntimeError("This route accepts only explicit non-final audio work")
    bound = {}
    for key in (("clock",) if voice_only else ("clock", "scorePlan", "wave")):
        spec = config[key]
        path = assembly.bound_path(ep, spec["path"], key)
        assembly.verify_sha(path, spec["sha256"], key)
        bound[key] = path
    clock = json.loads(bound["clock"].read_text())
    validate_working_clock(clock, config["episode"])
    if not voice_only:
        score = json.loads(bound["scorePlan"].read_text())
        if score.get("finalApproved") is not False or score.get("clockSha256") != config["clock"]["sha256"]:
            raise RuntimeError("Working score must bind this exact current clock")
    output = (ep / config["outputDirectory"]).resolve()
    if not output.is_relative_to(ep / "work") or output.name == "assembly-v1":
        raise RuntimeError("Working audio must stay in a separate work directory")
    output.mkdir(parents=True, exist_ok=True)
    plan = clock if voice_only else dict(clock, **{k: score[k] for k in ("scorePlacements", "musicBreaths", "musicNormalization")})
    voice = assembly.build_voice(ep, output, plan)
    outputs = {"voice": voice}
    if not voice_only:
        wave = json.loads(bound["wave"].read_text())
        outputs["music"] = assembly.build_music(ep, output, plan, wave)
    for key, path in bound.items():
        assembly.verify_sha(path, config[key]["sha256"], key)
    assembly.verify_sha(config_path,original_config_sha,'working stem config')
    receipt = {"kind": "working-audio-stems-not-final", "episode": config["episode"],
               "finalApproved": False, "renderablePictureCreated": False,
               "computeHost": socket.gethostname(), "configSha256": assembly.sha(config_path),
               "clockSha256": config["clock"]["sha256"], "voiceOnly": voice_only,
               "rendererSha256": assembly.sha(Path(assembly.__file__)),
               "runtimeSeconds": clock["runtimeSeconds"],
               "outputs": {k: {"path": str(p), "sha256": assembly.sha(p)} for k, p in outputs.items()}}
    (output / "audio-stem-audit.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
