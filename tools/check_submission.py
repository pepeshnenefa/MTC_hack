"""Offline packaging/CSV smoke check. Synthetic replies are NOT quality evaluation."""
import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'submission'))
from batch_io import read_questions, validate_answers


def check(archive, questions, kb, output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        with zipfile.ZipFile(archive) as package:
            names=package.namelist()
            expected={'solution.py','batch_io.py','retrieval.py','context.py','hybrid.py','hierarchy.py','requirements.txt'}
            if len(names)!=len(expected) or set(names)!=expected:
                raise ValueError('Unexpected or missing package entries')
            package.extractall(root)
        data=root/'data';data.mkdir()
        (data/'questions.csv').write_bytes(Path(questions).read_bytes())
        with zipfile.ZipFile(data/'knowledge-base.zip','w') as target:
            kb=Path(kb)
            for path in sorted(kb.rglob('*')):
                if path.is_file() and path.suffix.lower() in ('.md','.txt'):
                    target.write(path,path.relative_to(kb).as_posix())
        # Inject a local responder into the extracted solution, never ship this responder.
        harness=root/'offline_check.py'
        harness.write_text('''import os, sys, socket
from pathlib import Path
import solution

def forbidden(*args, **kwargs):
    raise AssertionError("Network forbidden in offline check")
socket.create_connection = forbidden
solution.urlopen = forbidden
class OfflineResponder:
    def __init__(self, **kwargs): pass
    def answer(self, question, sources):
        return 'ТЕСТ ФОРМАТА, не ответ модели. Источников: %s.\\nКавычки: "проверка".' % len(sources)
solution.API = OfflineResponder
os.environ['DATA_DIR'] = str(Path(__file__).parent / 'data')
solution.main()
''',encoding='utf-8')
        target=output/'format-check.csv'
        run=subprocess.run([sys.executable,str(harness),'--output',str(target)],cwd=root,
                           capture_output=True,text=True,encoding='utf-8',check=True)
        rows=read_questions(questions)
        count=validate_answers(target,rows)
        report={'kind':'offline-format-check','rows':count,'passed':True,
                'network_used':False,'model_quality_evaluated':False,
                'package_files':names,'runner_stdout':run.stdout.strip()}
        (output/'format-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,default=ROOT/'dist'/'energytool-submission.zip')
    parser.add_argument('--questions',type=Path,default=ROOT/'demo_questions.csv')
    parser.add_argument('--kb',type=Path,default=ROOT/'Knowledge-base')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'work'/'offline-check')
    args=parser.parse_args()
    print(json.dumps(check(args.archive,args.questions,args.kb,args.output_dir),ensure_ascii=False,indent=2))
