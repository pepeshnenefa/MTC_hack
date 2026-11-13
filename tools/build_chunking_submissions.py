"""Build three retrieval experiments from the exact semantic07 archive."""

import hashlib
import json
import zipfile

from build_grouped_context_submission import BASE, SHA, ROOT


MODES = ('paragraph', 'semantic', 'window')


def replace_once(data, before, after):
    if data.count(before) != 1:
        raise ValueError(f'Unexpected semantic07 source: {before!r}')
    return data.replace(before, after)


def build(mode):
    if mode not in MODES or hashlib.sha256(BASE.read_bytes()).hexdigest() != SHA:
        raise ValueError('Invalid mode or semantic07 archive')
    with zipfile.ZipFile(BASE) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files['chunking.py'] = (ROOT / 'submission/chunking.py').read_bytes()
    files['config.py'] = replace_once(
        files['config.py'], b"    'M_DENSE_MISMATCH':",
        f"    'CHUNK_MODE': {mode!r},\n    'M_DENSE_MISMATCH':".encode(),
    )
    files['solution.py'] = replace_once(
        files['solution.py'], b'from hybrid import prepare as prepare_hybrid',
        b'from chunking import prepare as prepare_hybrid',
    )
    files['retrieval.py'] = replace_once(
        files['retrieval.py'], b'candidates=deduplicate(candidates)[:k]',
        b'from chunking import restore_sections\n        candidates=deduplicate(restore_sections(candidates))[:k]',
    )
    output = ROOT / 'dist' / f'energytool-s07-chunk-{mode}.zip'
    manifest = output.with_suffix('.manifest.json')
    if output.exists() or manifest.exists():
        raise FileExistsError(output)
    for name, data in files.items():
        if '/' in name or '\\' in name or not (name.endswith('.py') or name == 'requirements.txt'):
            raise ValueError(name)
        if name.endswith('.py'):
            compile(data, name, 'exec')
    with zipfile.ZipFile(output, 'x') as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None or output.stat().st_size > 128_000_000:
            raise ValueError(f'Invalid archive: {output}')
    metadata = {
        'archive': output.name,
        'base_sha256': SHA,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'default_overrides': {'CHUNK_MODE': mode},
        'files': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    manifest.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    for mode in MODES:
        print(build(mode))
