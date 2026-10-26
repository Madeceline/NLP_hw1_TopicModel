"""Execute the walkthrough and persist outputs plus an audit manifest."""
from __future__ import annotations
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    root=Path(__file__).resolve().parents[1]
    path=root/'演示.ipynb'
    notebook=nbformat.read(path,as_version=4)
    manager=KernelManager(kernel_name='python3')
    # Ensure execution uses this exact environment, not another registered kernel.
    manager.kernel_spec.argv=[sys.executable,'-Xfrozen_modules=off','-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=900,
        resources={'metadata':{'path':str(root)}},allow_errors=False)
    client.on_cell_execute=lambda cell,cell_index:print(f'Execute code cell at index {cell_index}',flush=True)
    started=datetime.now(timezone.utc).isoformat()
    try:
        client.execute()
    finally:
        nbformat.write(notebook,path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    cells=[c for c in notebook.cells if c.cell_type=='code']
    errors=[{'cell':i,'ename':o.get('ename'),'evalue':o.get('evalue')}
        for i,c in enumerate(notebook.cells) for o in c.get('outputs',[]) if o.output_type=='error']
    manifest={'notebook':path.name,'started_at_utc':started,
        'finished_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,
        'total_cells':len(notebook.cells),'code_cells':len(cells),
        'executed_code_cells':sum(c.execution_count is not None for c in cells),
        'error_outputs':errors,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    output=root/'results/logs/notebook_execution.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['notebook','python']},indent=2))


if __name__=='__main__':main()
