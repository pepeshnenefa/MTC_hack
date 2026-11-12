"""Build a code-only ZIP from an explicit allowlist; no recursive packaging."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('solution.py', 'batch_io.py', 'config.py', 'retrieval.py', 'context.py', 'corrections.py', 'hybrid.py', 'hierarchy.py', 'checks.py', 'requirements.txt')


def with_defaults(source, overrides):
    """Variant archives differ only in config defaults (e.g. USE_HYBRID=0); the code is identical."""
    import re
    text=source.decode('utf-8')
    for name,value in overrides.items():
        pattern=re.compile(r"(    '"+re.escape(name)+r"': )([^,\n]+)(,)")
        if len(pattern.findall(text))!=1:raise ValueError('Unknown setting: '+name)
        literal=value if re.fullmatch(r'-?\d+(\.\d+)?',value) else repr(value)
        text=pattern.sub(lambda m:m.group(1)+literal+m.group(3),text)
    return text.encode('utf-8')


def build(output, overrides=None):
    output=Path(output)
    contents={name:(ROOT/'submission'/name).read_bytes() for name in FILES}
    if overrides:contents['config.py']=with_defaults(contents['config.py'],overrides)
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
              'default_overrides':overrides or {},
              'files':{n:hashlib.sha256(b).hexdigest() for n,b in contents.items()}}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'dist'/'energytool-submission.zip')
    parser.add_argument('--set',action='append',default=[],help='NAME=VALUE default for a variant archive')
    args=parser.parse_args()
    print(json.dumps(build(args.output,dict(pair.split('=',1) for pair in args.set)),indent=2))
