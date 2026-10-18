"""Verify a finite H3 queue, ingest silent picture, and create diagnostic sheets.

Without flags this runs on the qualified GPU worker and writes local-audit.json.
--preview-sheets draws the same sampled sheets from the hash-checked raw
outputs anywhere ffmpeg runs, into review-preview-not-final/, and never
ingests or writes local-audit.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common
from prepare_episode_production import ingest, probe

PREVIEW = "-preview-not-final"


if not __debug__:
    raise SystemExit("Optimized Python disables production safety gates")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sheet_ordinals(frames):
    """Every 12th native frame plus the exact final frame."""
    return sorted(set(range(0, frames, 12)) | {frames - 1})


def draw_sheet(source, frames, sheet):
    """The sampled review sheet: a 5-wide grid of every 12th frame and the last."""
    ordinals = sheet_ordinals(frames)
    tile_rows = (len(ordinals)+4)//5
    selection = f"select='eq(mod(n,12),0)+eq(n,{frames-1})'"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source),
        "-vf", f"{selection},scale=448:-1,tile=5x{tile_rows}",
        "-frames:v", "1", str(sheet)], check=True)
    return ordinals


def preview_sheets(root, inputs_dir="inputs", outputs_dir="outputs"):
    """Sheets from completed raw outputs on any host; labelled, never admitted."""
    manifest_path = (root/inputs_dir/"manifest.json").resolve()
    manifest_path.relative_to(root)
    manifest = json.loads(manifest_path.read_text())
    output_root = (root/outputs_dir).resolve()
    output_root.relative_to(root)
    state_bytes = (output_root/"state.json").read_bytes()
    state = json.loads(state_bytes)
    if state["manifestSha256"] != sha(manifest_path):
        raise ValueError("Queue state belongs to another manifest")
    host = _common.preview_host_record(None)
    dest = root/("review" + PREVIEW)
    dest.mkdir(exist_ok=True)
    rows = []
    for item in manifest["items"]:
        rec = state["jobs"].get(item["id"], {})
        if rec.get("status") != "complete":
            rows.append(dict(id=item["id"], status="not-complete"))
            continue
        raw = (output_root/rec["output"]).resolve()
        raw.relative_to(output_root)
        if sha(raw) != rec["sha256"] or rec["sourceSha256"] != item["sourceSha256"]:
            raise ValueError("Raw output or source changed: " + item["id"])
        counted = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "json", str(raw)],
            check=True, capture_output=True, text=True).stdout)
        frames = int(counted["streams"][0]["nb_read_frames"])
        sheet = dest/(item["id"] + "-2fps" + PREVIEW + ".jpg")
        ordinals = draw_sheet(raw, frames, sheet)
        rows.append(dict(id=item["id"], status="sheet-drawn", rawSha256=rec["sha256"], frames=frames,
                         expectedFrames=item.get("expectedFrames"), sheet=sheet.name, sheetSha256=sha(sheet),
                         frameOrdinals=ordinals, timestamps=[round(n/24, 6) for n in ordinals]))
    audit = dict(kind="h3-review-sheets-preview-not-final", manifestSha256=sha(manifest_path),
                 stateSha256=hashlib.sha256(state_bytes).hexdigest(), hostRecord=host, finalApproved=False,
                 reviewerSourceSha256=sha(Path(__file__)),
                 scope="Sampled sheets of raw outputs for looking only; no ingest, no technical pass, no admission",
                 outputs=rows)
    _common.atomic_json(dest/"preview-sheet-audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "outputs"}))
    return audit


def review_used_tails(root, inputs_dir="inputs", tail_plan=None):
    """Separate immutable-source audit of the exact proposed continuation windows."""
    if platform.system() != "Linux":
        raise RuntimeError("Native-frame review belongs on the qualified GPU worker")
    subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],check=True,capture_output=True)
    manifest_path=root/inputs_dir/"manifest.json"; manifest=json.loads(manifest_path.read_text())
    audit_path=root/"local-audit.json"; original_audit=audit_path.read_bytes(); audit=json.loads(original_audit)
    assert audit['manifestSha256']==sha(manifest_path)
    outputs={r['id']:r for r in audit['outputs'] if r['status']=='technical-pass'}
    planned = {}
    plan_digest = None
    if tail_plan is not None:
        tail_plan = tail_plan.resolve()
        tail_plan.relative_to(root)
        plan = json.loads(tail_plan.read_text())
        assert plan['manifestSha256'] == sha(manifest_path)
        windows = plan['windows']
        assert windows and len({x['id'] for x in windows}) == len(windows)
        assert {x['id'] for x in windows}.issubset({x['id'] for x in manifest['items']})
        assert all(type(x['frames']) is int and x['frames'] > 0 for x in windows)
        planned = {x['id']: x['frames'] for x in windows}
        plan_digest = sha(tail_plan)
    dest=root/'tail-review';dest.mkdir(exist_ok=True)
    def one(item):
        last=planned.get(item['id'],item.get('plannedCoverageFramesThisWave',item.get('exactMeasuredDeficitFrames')))
        if last is None or item['id'] not in outputs:
            return None
        row=outputs[item['id']]; source=Path(row['silent']['file'])
        assert sha(source)==row['silent']['sha256'] and 1<=last<row['silent']['frames']
        ordinals=sorted(set([0,1,last])|set(range(0,last+1,1 if last<=24 else 4)))
        columns=min(5,len(ordinals));height=(len(ordinals)+columns-1)//columns
        select="select='"+"+".join(f'eq(n,{n})' for n in ordinals)+"'"
        sheet=dest/(item['id']+'-used-tail.jpg')
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-vf',
                        f'{select},scale=448:-1,tile={columns}x{height}','-frames:v','1',str(sheet)],check=True)
        return dict(id=item['id'],silentSha256=row['silent']['sha256'],anchorSha256=item['sourceSha256'],
                    usedWindow=dict(firstFrame=1,lastFrameInclusive=last,frames=last),
                    conditioningFrameIncludedForComparisonOnly=0,frameOrdinals=ordinals,
                    sheet=str(sheet),sheetSha256=sha(sheet),verdict='pending-sampled-review')
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=[r for r in pool.map(one,manifest['items']) if r]
    assert audit_path.read_bytes()==original_audit
    if tail_plan is not None:
        assert sha(tail_plan) == plan_digest
    result=dict(manifestSha256=sha(manifest_path),baseAuditSha256=hashlib.sha256(original_audit).hexdigest(),
                computeHost=platform.node(),reviewerSha256=sha(Path(__file__)),outputs=rows,tailPlanSha256=plan_digest,
                scope='Exact proposed used windows, all frames for tails up to24frames and every fourth plus endpoints otherwise; not continuous-watch certification')
    (root/'tail-window-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=len(rows),manifestSha256=result['manifestSha256'])))


def main(root, inputs_dir="inputs", outputs_dir="outputs"):
    if platform.system() != "Linux":
        raise RuntimeError("Media ingest/decode/review sheets run on the qualified GPU worker, not the Mac")
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                   check=True, capture_output=True)
    manifest_path = (root/inputs_dir/"manifest.json").resolve()
    manifest_path.relative_to(root)
    manifest = json.loads(manifest_path.read_text())
    output_root = (root/outputs_dir).resolve()
    output_root.relative_to(root)
    state_bytes = (output_root/"state.json").read_bytes()
    state_digest = hashlib.sha256(state_bytes).hexdigest()
    state = json.loads(state_bytes)
    assert state["manifestSha256"] == sha(manifest_path)
    grouped_path = root / 'silent' / 'ingest-receipt.json'
    grouped = {}
    if grouped_path.exists():
        receipt = json.loads(grouped_path.read_text())
        assert receipt['sourceStateSha256'] == state_digest, 'Changed ingest source state'
        grouped = {r['id']: r for r in receipt['items']}
        assert len(grouped) == len(receipt['items']), 'Duplicate ingest receipt ID'
    sheet_root = root/"review"
    sheet_root.mkdir(exist_ok=True)
    snapshot = sheet_root / f"state-snapshot-{state_digest[:12]}.json"
    if snapshot.exists():
        assert snapshot.read_bytes() == state_bytes
    else:
        snapshot.write_bytes(state_bytes)
    def one(item):
        rec = state["jobs"].get(item["id"], {})
        if rec.get("status") != "complete":
            return dict(id=item["id"], status="not-complete")
        raw = output_root/rec["output"]
        assert sha(raw) == rec["sha256"]
        assert rec["sourceSha256"] == item["sourceSha256"]
        if item['id'] in grouped:
            rec_ingest = grouped[item['id']]
            target = root / 'silent' / rec_ingest['silent']
            assert target.parent == root / 'silent', 'Unsafe silent output path'
            assert rec_ingest['rawSha256'] == rec['sha256']
            assert sha(target) == rec_ingest['silentSha256'], 'Changed silent output'
            media = probe(target)
            assert len(media['streams']) == 1
            video = media['streams'][0]
            assert video['codec_type'] == 'video' and rec_ingest['audioStreams'] == 0
            assert int(video['nb_frames']) == rec_ingest['frames']
            silent = dict(file=str(target), sha256=rec_ingest['silentSha256'],
                          frames=rec_ingest['frames'], audioStreams=0,
                          width=video['width'], height=video['height'],
                          fps=video['r_frame_rate'], duration=float(video['duration']),
                          sourceSha256=rec['sha256'], ingestReceiptSha256=sha(grouped_path))
        else:
            silent = ingest((raw, root/"silent"/(item["id"]+".mp4")))
        assert silent["frames"] == item["expectedFrames"] and silent["audioStreams"] == 0
        sheet = sheet_root/(item["id"]+"-2fps.jpg")
        ordinals = draw_sheet(silent["file"], silent["frames"], sheet)
        return dict(id=item["id"], status="technical-pass", rawSha256=rec["sha256"],
                    sourceSha256=rec["sourceSha256"], silent=silent, sheet=str(sheet),
                    sheetSha256=sha(sheet), frameOrdinals=ordinals,
                    timestamps=[round(n/24,6) for n in ordinals],
                    creativeVerdict="pending-independent-review")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, manifest["items"]))
    count = sum(r["status"]=="technical-pass" for r in rows)
    audit = dict(expected=len(rows), complete=count, technicalPass=count==len(rows),
                 manifestSha256=sha(manifest_path), stateSha256=state_digest,
                 stateSnapshot=str(snapshot),
                 stateFileChangedDuringReview=sha(output_root/"state.json") != state_digest,
                 outputsDirectory=str(output_root),
                 reviewerSourceSha256=sha(Path(__file__)),
                 computeHost=platform.node(),
                 frameSampling="Every 12th native frame plus exact endpoint; row-major grid, not continuous temporal certification",
                 outputs=rows)
    (root/"local-audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({k:v for k,v in audit.items() if k!="outputs"}))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--inputs-dir", default="inputs")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--tail-windows-only", action="store_true")
    p.add_argument("--tail-plan", type=Path, help="Separate hash-bound used-window plan; generation manifest remains immutable")
    p.add_argument("--preview-sheets", action="store_true",
                   help="Draw labelled sheets from raw outputs anywhere ffmpeg runs; no ingest, no local-audit.json")
    args = p.parse_args()
    if args.preview_sheets:
        preview_sheets(args.root.resolve(), args.inputs_dir, args.outputs_dir)
    elif args.tail_windows_only:
        review_used_tails(args.root.resolve(), args.inputs_dir, args.tail_plan)
    else:
        main(args.root.resolve(), args.inputs_dir, args.outputs_dir)
