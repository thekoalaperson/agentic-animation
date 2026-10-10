"""Render the Hindi Short preset on a Linux GPU host with complex libass shaping.

The qualified Devanagari and Romanized text modes record font hashes, shaping
logs and safe-area bounds. Actual output review remains separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal(text):
    if any(x in text for x in ('{', '}', '\\')):
        raise ValueError('ASS controls are forbidden in display text')
    return text.replace('\n', r'\N')


def event(text, size, y, font='Noto Serif Devanagari', color='&H008AD9F4', bold=-1):
    return dict(text=text, size=size, y=y, font=font, color=color, bold=bold)


def script(rows):
    head = '''[Script Info]
ScriptType: v4.00+
PlayResX: 2160
PlayResY: 3840
ScaledBorderAndShadow: yes
WrapStyle: 0
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
'''
    for i, r in enumerate(rows):
        # Noto's larger Devanagari em metrics otherwise make a nominal58px
        # visibly smaller than the house Latin type. Qualify the glyph result.
        em_scale=1.5 if r['font']=='Noto Serif Devanagari' else 1
        head += (f"Style: s{i},{r['font']},{r['size']*2*em_scale},{r['color']},&H000000FF,"
                 f"&H00000000,&H10000000,{r['bold']},0,0,0,100,100,0,0,1,0,8,8,344,344,0,1\n")
    head += '[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n'
    for i, r in enumerate(rows):
        head += (f"Dialogue: 0,0:00:00.00,0:00:01.00,s{i},,0,0,0,,"
                 + '{' + f"\\an8\\pos(1080,{r['y']*2})" + '}' + literal(r['text'])+'\n')
    return head


def alpha_bounds(path, width=2160, height=3840):
    raw = subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-frames:v','1',
        '-pix_fmt','rgba','-f','rawvideo','-'])
    pixels = np.frombuffer(raw,np.uint8).reshape(height,width,4)
    ys,xs = np.where(pixels[:,:,3] > 4)
    if not len(xs): raise ValueError('Empty text raster')
    return [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]


def raster(mode, output, width, height, text, *, fonts_dir=None,
           channel_name=None, channel_handle=None, latin_font='Noto Serif',
           devanagari_font='Noto Serif Devanagari'):
    if platform.system()!='Linux' or (width,height)!=(2160,3840):
        raise RuntimeError('2x typography requires the configured Linux GPU host')
    subprocess.run(['nvidia-smi','-L'],check=True,capture_output=True)
    fonts_dir = fonts_dir or os.environ.get('AGENTIC_TYPE_FONTS')
    if not fonts_dir:
        raise ValueError('Provide fonts_dir or set AGENTIC_TYPE_FONTS')
    fonts=Path(fonts_dir).expanduser().resolve()
    font_files=sorted(p for p in fonts.iterdir() if p.suffix.lower() in ('.ttf','.otf','.ttc'))
    if not font_files: raise ValueError('Explicit retained font directory required')
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise ValueError('Preserve completed typography')
    if mode not in ('caption-safe','short-question','channel-cta'):
        raise ValueError('Unsupported bounded Short text mode')
    if mode == 'channel-cta' and (not isinstance(channel_name, str) or not channel_name.strip()
            or not isinstance(channel_handle, str) or not re.fullmatch(r'@[^\s/]+', channel_handle)):
        raise ValueError('Channel CTA requires explicit channel_name and channel_handle')
    for font in (latin_font, devanagari_font):
        if not font or any(c in font for c in ',\r\n'):
            raise ValueError('Invalid font family')
    attempts=[]
    for size in range(58,43,-1):
        if mode=='channel-cta':
            rows=[event(channel_name,30,450,font=latin_font),
                  event('FULL VIDEO\nON THE CHANNEL',64,610,font=latin_font,color='&H00E0EDF0'),
                  event(text,48,970,font=devanagari_font),
                  event(channel_handle,36,1160,font=latin_font,color='&H00E0EDF0',bold=0)]
        else:
            rows=[event(text,size,1300 if mode=='caption-safe' else 420,
                        font=devanagari_font if any('\u0900'<=c<='\u097f' for c in text) else latin_font)]
        ass=output.with_suffix('.ass');ass.write_text(script(rows),encoding='utf8')
        for path in (ass,fonts):
            if any(c in str(path) for c in "':,[]\\"):
                raise ValueError('Unsafe filter path')
        color='0x090c12' if mode=='channel-cta' else 'black@0'
        cmd=['ffmpeg','-y','-v','info','-f','lavfi','-i',
             f'color=c={color}:s=2160x3840:r=1,format=rgba',
             '-vf',f'ass=filename={ass}:fontsdir={fonts}:alpha=1:shaping=complex',
             '-frames:v','1','-threads','2',str(output)]
        run=subprocess.run(cmd,check=True,capture_output=True,text=True)
        logs=run.stderr
        # Retain the evidence even when a missing font/geometry gate fails.
        output.with_suffix('.log').write_text(logs)
        if 'HarfBuzz' not in logs or '(COMPLEX)' not in logs:
            raise ValueError('Complex script shaper not evidenced')
        if 'Glyph 0x' in logs or 'fontselect: failed' in logs:
            raise ValueError('Missing glyph or unintended font fallback; inspect renderer log')
        # CTA is opaque; validate its text geometry on a separately transparent pass.
        geometry=output
        if mode=='channel-cta':
            geometry=output.with_name(output.stem+'-geometry.png')
            gc=cmd.copy();gc[gc.index('-i')+1]='color=c=black@0:s=2160x3840:r=1,format=rgba';gc[-1]=str(geometry)
            subprocess.run(gc,check=True,capture_output=True)
        bounds=alpha_bounds(geometry)
        x0,y0,x1,y1=bounds
        safe=(x0>=300 and x1<=1860 and y0>=780 and y1<=3080)
        if mode=='caption-safe':safe=safe and y1<=3048
        if mode=='short-question':safe=safe and y1<=1320
        attempts.append(dict(fontSize=size,bounds2x=bounds,passed=safe))
        if safe:break
        if mode=='channel-cta':raise ValueError('CTA text exceeds safe area')
    else:raise ValueError('Caption cannot fit at the allowed house sizes')
    output.with_suffix('.log').write_text(logs)
    report=dict(kind='gpu-host-complex-shaped-house-type',mode=mode,text=text,
        dimensions=[width,height],outputSha256=sha(output),scriptSha256=sha(ass),
        fonts=[dict(path=str(p),sha256=sha(p)) for p in font_files],
        shaper='libass complex/HarfBuzz, evidenced in retained log',
        bounds2x=bounds,attempts=attempts,computeHost=platform.node(),
        sourceSha256=sha(__file__),safeAreaPassed=True,visualApproval=False,
        policy='Typography runs on the configured GPU host; visual approval is recorded separately')
    output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def thumbnail_recipe(recipe_path, output_dir):
    """Exact Latin-letter house thumbnails, rendered at2x on the GPU host."""
    if platform.system() != 'Linux':
        raise RuntimeError('Thumbnail typography requires the configured Linux GPU host')
    subprocess.run(['nvidia-smi', '-L'], check=True, capture_output=True)
    recipe_path = Path(recipe_path).resolve()
    recipe = json.loads(recipe_path.read_text())
    root = (recipe_path.parent / recipe['episodeRoot']).resolve()
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(root / 'work'):
        raise ValueError('Thumbnail work output escapes the episode')
    if recipe['renderSize'] != [2560,1440] or recipe['exportSize'] != [1280,720]:
        raise ValueError('House thumbnail geometry required')
    font = Path(recipe['font']).expanduser()
    font = (font if font.is_absolute() else recipe_path.parent / font).resolve()
    if not font.is_file():
        raise ValueError('Retained font missing')
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for entry in recipe['entries']:
        key = entry['id']
        if not re.fullmatch(r'[a-z0-9-]+', key):
            raise ValueError('Unsafe variant ID')
        source = (root / entry['source']).resolve()
        if not source.is_relative_to(root) or sha(source) != entry['sourceSha256']:
            raise ValueError('Changed thumbnail plate')
        filters = []
        for i,line in enumerate(entry['lines']):
            if not line['text'].strip() or not re.fullmatch(r'[A-Za-z0-9 !?.-]+', line['text']):
                raise ValueError('This recipe accepts exact Romanized text only')
            if not re.fullmatch(r'(white|0x[0-9a-fA-F]{6})', line['color']):
                raise ValueError('Invalid text color')
            textfile = output_dir / f'{key}-line-{i}.txt'
            textfile.write_text(line['text'])
            for path in (font,textfile):
                if any(c in str(path) for c in "':,[]\\"):
                    raise ValueError('Unsafe text filter path')
            x,y,size = (int(line[k]) for k in ('x','y','size'))
            if not 80 <= size <= 220 or not 24 <= x < 2400 or not 24 <= y < 1300:
                raise ValueError('Unbounded thumbnail text position/size')
            filters.append(f'drawtext=fontfile={font}:textfile={textfile}:x={x}:y={y}:fontsize={size}:fontcolor={line["color"]}:borderw=10:bordercolor=0x101321:shadowx=4:shadowy=5:shadowcolor=black@0.6')
        overlay = output_dir / f'{key}-type.png'
        command = ['ffmpeg','-v','info','-n','-f','lavfi','-i','color=c=black@0:s=2560x1440:r=1,format=rgba','-vf',','.join(filters),'-frames:v','1','-threads','2',str(overlay)]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        (output_dir / f'{key}-type.log').write_text(result.stderr)
        bounds = alpha_bounds(overlay,2560,1440)
        if not (bounds[0]>=24 and bounds[1]>=24 and bounds[2]<=2536 and bounds[3]<=1416):
            raise ValueError('Thumbnail text geometry not verified')
        master = output_dir / f'thumb-{key}-2x.png'
        graph = '[0:v]scale=2560:1440:force_original_aspect_ratio=increase,crop=2560:1440[base];[base][1:v]overlay=0:0[out]'
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(source),'-i',str(overlay),'-filter_complex',graph,'-map','[out]','-frames:v','1','-threads','2',str(master)],check=True,capture_output=True)
        png = output_dir / f'thumb-{key}-1280.png'
        jpg = output_dir / f'thumb-{key}-1280.jpg'
        mobile = output_dir / f'thumb-{key}-mobile.jpg'
        for target,width,quality in [(png,1280,[]),(jpg,1280,['-q:v','2']),(mobile,320,['-q:v','3'])]:
            subprocess.run(['ffmpeg','-v','error','-n','-i',str(master),'-vf',f'scale={width}:-1:flags=lanczos','-frames:v','1','-threads','2',*quality,str(target)],check=True,capture_output=True)
        if jpg.stat().st_size > 2_000_000:
            raise ValueError('Upload JPEG exceeds the house limit')
        records.append(dict(id=key,sourceSha256=sha(source),lines=entry['lines'],textBounds2x=bounds,
            files={p.name:sha(p) for p in (master,png,jpg,mobile)},jpegBytes=jpg.stat().st_size))
    report = dict(method='GPU-host FreeType exact Romanized text,2x then Lanczos downsample',
                  recipeSha256=sha(recipe_path),font=dict(path=str(font),sha256=sha(font)),
                  sourceSha256=sha(__file__),variants=records,finalVisualApproval=False)
    (output_dir/'render-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode');ap.add_argument('output');ap.add_argument('text')
    ap.add_argument('--fonts',type=Path)
    ap.add_argument('--channel-name');ap.add_argument('--channel-handle')
    ap.add_argument('--latin-font',default='Noto Serif')
    ap.add_argument('--devanagari-font',default='Noto Serif Devanagari')
    args=ap.parse_args()
    result = thumbnail_recipe(args.text,args.output) if args.mode=='thumbnail-recipe' else raster(
        args.mode,args.output,2160,3840,args.text,fonts_dir=args.fonts,
        channel_name=args.channel_name,channel_handle=args.channel_handle,
        latin_font=args.latin_font,devanagari_font=args.devanagari_font)
    print(json.dumps(result,ensure_ascii=False))
