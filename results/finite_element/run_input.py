from pathlib import Path
import argparse, subprocess
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--abaqus',default='abaqus');p.add_argument('--cpus',type=int,default=4)
a=p.parse_args();inp=Path(a.input).resolve()
if not inp.is_file() or inp.suffix.lower()!='.inp': raise SystemExit('An existing INP is required.')
subprocess.run([a.abaqus,'job='+inp.stem,'input='+inp.name,'cpus='+str(a.cpus),'double=both','interactive'],cwd=str(inp.parent),check=True,shell=True)
