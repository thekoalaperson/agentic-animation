#!/usr/bin/env python3
"""Render native-resolution end-card typography on a configured Linux GPU host."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess


def font_path(recipe, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(recipe).resolve().parent / path).resolve()


def overlay_title(cfg, recipe, source, output, width, expected_sha):
    """Render authored type on one native-resolution moving shot, not a resized master."""
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if cfg.get('kind') != 'native-opening-title-overlay' or digest(source) != expected_sha:
        raise ValueError('Exact opening-title recipe and source hash required')
    if output.exists():
        raise ValueError('Preserve earlier title takes; choose a new output')
    streams = json.loads(subprocess.check_output([
        'ffprobe','-v','error','-show_streams','-of','json',str(source)], text=True))['streams']
    height = width*9//16
    if (len(streams) != 1 or streams[0]['codec_type'] != 'video'
            or (streams[0]['width'], streams[0]['height'], streams[0]['avg_frame_rate'])
            != (width, height, '24/1')):
        raise ValueError('Silent native-resolution24fps picture source required')
    frames = int(streams[0]['nb_frames'])
    timing = [cfg.get(k) for k in ('startSeconds','endSeconds','fadeInSeconds','fadeOutSeconds')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in timing):
        raise ValueError('Finite authored title timing required')
    start,end,fade_in,fade_out = timing
    if not (0 <= start < end <= frames/24 and 0 < fade_in and 0 < fade_out
            and fade_in+fade_out < end-start):
        raise ValueError('Title fades exceed the source shot')
    alpha = f'if(lt(t,{start}),0,if(lt(t,{start+fade_in}),(t-{start})/{fade_in},if(lt(t,{end-fade_out}),1,if(lt(t,{end}),({end}-t)/{fade_out},0))))'
    filters = []
    for i,row in enumerate(cfg['lines']):
        font = font_path(recipe, cfg['fonts'][row['font']])
        text_path = output.parent/f'{output.stem}-line-{i}.txt'
        text_path.write_text(row['text'], encoding='utf-8')
        if not font.is_file() or any(c in str(font)+str(text_path.resolve()) for c in "':,[]\\"):
            raise ValueError('Missing font or unsafe filter path')
        filters.append(f"drawtext=fontfile={font}:textfile={text_path.resolve()}:expansion=none:text_shaping=1:fontsize={round(row['size']*height)}:fontcolor={row['color']}:x={round(row['x']*width)}:y={round(row['y']*height)}:borderw={max(1,round(height/360))}:bordercolor=0x142039:shadowx={round(width/640)}:shadowy={round(height/240)}:shadowcolor=0x0B172D@0.8:alpha='{alpha}'")
    subprocess.run(['ffmpeg','-n','-v','error','-i',str(source),'-map','0:v:0','-an',
                    '-vf',','.join(filters),'-frames:v',str(frames),'-r','24','-c:v','libx264',
                    '-preset','slow','-crf','17','-pix_fmt','yuv420p','-threads','4',str(output)],check=True)
    final = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(output)],text=True))['streams']
    if len(final)!=1 or int(final[0]['nb_frames'])!=frames or final[0]['avg_frame_rate']!='24/1':
        raise ValueError('Title overlay changed the picture clock')
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(output),'-f','null','-'],check=True)
    if digest(source)!=expected_sha:
        raise ValueError('Opening source changed during rendering')
    audit = {'kind':'native-moving-opening-title','source':str(source),'sourceSha256':expected_sha,
             'outputSha256':digest(output),'recipeSha256':digest(recipe),'frames':frames,
             'resolution':[width,height],'fps':24,'decode':'passed','visualApproval':False,
             'lines':cfg['lines'],'timing':dict(zip(('start','end','fadeIn','fadeOut'),timing))}
    output.with_suffix('.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False),flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recipe', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--width', type=int, required=True)
    ap.add_argument('--video', type=Path, help='Optional silent native opening shot for moving title overlay')
    ap.add_argument('--source-sha256', help='Required immutable source binding with --video')
    a = ap.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('Production graphics rendering requires the configured Linux GPU host')
    subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], check=True)
    if a.width not in (1280, 1920): raise ValueError('Only native720p/1080p cards')
    cfg = json.loads(a.recipe.read_text())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.video:
        overlay_title(cfg, a.recipe, a.video.resolve(), a.output.resolve(), a.width, a.source_sha256)
        return
    width, height = a.width, a.width*9//16
    filters = []
    for i, row in enumerate(cfg['lines']):
        # textfile + expansion=none makes punctuation/content data, never filter syntax.
        font = font_path(a.recipe, cfg['fonts'][row['font']])
        if not font.is_file(): raise ValueError(f'Missing exact font: {font}')
        text_path = a.output.parent / f'{a.output.stem}-line-{i}.txt'
        text_path.write_text(row['text'], encoding='utf-8')
        if any(c in str(font)+str(text_path.resolve()) for c in "':,[]\\"):
            raise ValueError('Unsafe filter path')
        filters.append(f"drawtext=fontfile={font}:textfile={text_path.resolve()}:expansion=none:fontsize={round(row['size']*height)}:fontcolor={row['color']}:x=(w-text_w)/2:y={round(row['y']*height)}")
    filters.append(f"drawbox=x={round(width*.44)}:y={round(height*.54)}:w={round(width*.12)}:h=2:color=0x827065:t=fill")
    command = ['ffmpeg','-y','-v','error','-f','lavfi','-i',
               f"color=c={cfg['background']}:s={width}x{height}:r=24",
               '-vf', ','.join(filters), '-frames:v','1','-threads','2',str(a.output)]
    subprocess.run(command, check=True)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit = {'kind':'native-house-end-card', 'resolution':[width,height],
             'recipeSha256':digest(a.recipe), 'outputSha256':digest(a.output),
             'fonts':{k:{'path':str(font_path(a.recipe,v)),'sha256':digest(font_path(a.recipe,v))} for k,v in cfg['fonts'].items()},
             'lines':cfg['lines'], 'host':platform.node(), 'visualApproval':False,
             'note':'Typography encoded on the configured GPU host. No generated imagery.'}
    a.output.with_suffix('.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__': main()
