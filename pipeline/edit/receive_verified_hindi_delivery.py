"""Receive a GPU-verified Hindi package without repeating media compute locally.

The trusted remote manifest hash, source-bound configuration, exact media bytes,
and all support files are mandatory. This does not create new media QA evidence.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
try:
    from . import package_hindi_delivery as delivery
except ImportError:
    import package_hindi_delivery as delivery


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('episode','config','support','receipt'):ap.add_argument(name,type=Path)
    ap.add_argument('--manifest-sha256',required=True)
    a=ap.parse_args();ep=a.episode.resolve();config=a.config.resolve()
    receipt=delivery.episode_path(ep,str(a.receipt),'receipt')
    receipt.relative_to(ep/'work')
    assert not receipt.exists(),'Preserve reception evidence'
    manifest_path=a.support/'FINAL_MANIFEST.json'
    assert delivery.sha256_file(manifest_path)==a.manifest_sha256,'Remote manifest identity mismatch'
    cfg,gates=delivery.analyze_config(ep,config);assert not gates,gates
    doc=json.loads(manifest_path.read_text())
    assert doc['provenance']['configSha256']==delivery.sha256_file(config)
    assert doc['publishingAuthorized'] is False
    for key in delivery.MEDIA_ASSET_IDS:
        row=doc['validation']['media'][key]
        assert row['fullDecode']=='passed' and row['gpuDecode']['passed']
        assert row['gpuDecode']['sourceSha256']==cfg['assets'][key]['sha256']
    expected=delivery._expected_output_paths(cfg)
    by_id={r['id']:r for r in doc['assets']}
    assert set(by_id)==set(expected)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.received-final-',dir=receipt.parent) as td:
        candidate=Path(td)/'deliverables';candidate.mkdir()
        shutil.copy2(manifest_path,candidate/'FINAL_MANIFEST.json')
        for key,relative in expected.items():
            record=by_id[key];assert record['canonical']=='deliverables/'+relative
            source=(delivery.episode_path(ep,cfg['assets'][key]['source'],key,must_exist=True)
                    if key in delivery.MEDIA_ASSET_IDS else (a.support/relative).resolve())
            if key not in delivery.MEDIA_ASSET_IDS:source.relative_to(a.support.resolve())
            assert source.is_file() and not source.is_symlink()
            assert source.stat().st_size==record['bytes']
            assert delivery.sha256_file(source)==record['sha256']
            output=candidate/relative;output.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,output)
            assert delivery.sha256_file(output)==record['sha256']
        checked=delivery.validate_package_structure(cfg,candidate)
        assert checked['fileCount']==len(expected)+1
        destination=ep/'deliverables'
        with delivery._exclusive_promotion_lock(ep):
            if destination.exists():
                assert destination.is_dir() and delivery._tree_hashes(destination)==delivery._tree_hashes(candidate),'Refusing existing nonidentical delivery'
            else:delivery._rename_noreplace(candidate,destination)
    result={'kind':'byte-exact-reception-of-gpu-verified-delivery',
            'completedAtUTC':datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            'episode':ep.name,'manifestSha256':a.manifest_sha256,
            'configSha256':delivery.sha256_file(config),'files':len(expected)+1,
            'localMediaProcessingPerformed':False,'freshMediaDecodeClaimed':False,
            'allAssetBytesMatchGpuManifest':True,'localStructureAndKitChecks':checked,
            'sourcesDeleted':False,'publishingAuthorized':False}
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
