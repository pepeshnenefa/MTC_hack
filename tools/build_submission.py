"""Build a code-only ZIP from an explicit allowlist; no recursive packaging."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('solution.py', 'batch_io.py', 'retrieval.py', 'requirements.txt')


def build(output):
    output=Path(output)
    contents={name:(ROOT/'submission'/name).read_bytes() for name in FILES}
    for name, data in contents.items():
        if name.endswith('.py'):
            compile(data.decode('utf-8'),name,'exec')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in contents.items():
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            archive.writestr(info,data)
    if output.stat().st_size > 128_000_000:
        raise ValueError('ZIP exceeds 128 MB')
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist())==set(FILES)
        assert archive.testzip() is None
    manifest={'archive':output.name,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'files':{n:hashlib.sha256(b).hexdigest() for n,b in contents.items()}}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'dist'/'energytool-submission.zip')
    args=parser.parse_args()
    print(json.dumps(build(args.output),indent=2))
