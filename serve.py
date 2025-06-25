from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated
import concurrent
import traceback
import asyncio
import uuid
import os

import sys

from rich.logging import RichHandler
import logging as L 

L.shutdown()
L.getLogger('stanza').handlers.clear()
L.getLogger('transformers').handlers.clear()
L.getLogger('nemo_logger').handlers.clear()
L.getLogger("stanza").setLevel(L.INFO)
L.getLogger('nemo_logger').setLevel(L.CRITICAL)
L.getLogger('batchalign').setLevel(L.WARN)
L.getLogger('lightning.pytorch.utilities.migration.utils').setLevel(L.ERROR)
L.basicConfig(format="%(message)s", level=L.ERROR, handlers=[RichHandler(rich_tracebacks=True)])
L.getLogger('nemo_logger').setLevel(L.INFO)
L.getLogger('batchalign').setLevel(L.INFO)
L = L.getLogger('batchalign')

from batchalign import BatchalignPipeline, CHATFile

from pathlib import Path
WORKDIR = Path(os.getenv("BA2_WORKDIR", ""))
WORKDIR.mkdir(exist_ok=True)

app = FastAPI(title="TalkBank | Batchalign2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from batchalign.pipelines.dispatch import WhisperXEngine, NemoSpeakerEngine
import tempfile
from fastapi import File

# Engines will be initialized per request in the transcribe endpoint

# Load WhisperXEngine globally at startup
whisperx_engine = WhisperXEngine(lang="eng")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def upload_form():
    return """
    <h1>Batchalign2 Transcription</h1>
    <form action="/transcribe" enctype="multipart/form-data" method="post">
      <label for="file">Upload audio/video file (.wav, .mp3, .mp4):</label><br>
      <input type="file" id="file" name="file" accept=".wav,.mp3,.mp4" required><br><br>
      <label for="num_speakers">Number of speakers:</label>
      <input type="number" id="num_speakers" name="num_speakers" min="1" max="10" value="1" required><br><br>
      <button type="submit">Transcribe</button>
    </form>
    """
    
@app.post("/transcribe", response_class=HTMLResponse, include_in_schema=False)
async def transcribe(file: UploadFile = File(...), num_speakers: int = Form(...)):
    # Save uploaded file to a local uploads directory under the project
    from pathlib import Path
    import shutil

    base_upload_dir = Path("uploads/tmp")
    base_upload_dir.mkdir(parents=True, exist_ok=True)
    request_id = str(uuid.uuid4())
    request_dir = base_upload_dir / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    # Create a timestamped filename for the upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    input_path = request_dir / safe_filename
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Initialize NemoSpeakerEngine per request (num_speakers is dynamic)
    nemo_speaker_engine = NemoSpeakerEngine(num_speakers=num_speakers)

    # Create pipeline with preloaded engines
    pipeline = BatchalignPipeline(whisperx_engine, nemo_speaker_engine)

    # Run pipeline on input file path
    doc = pipeline(str(input_path), num_speakers=num_speakers, asr="whisperx")

    # Write output .cha file
    output_basename = safe_filename.rsplit('.', 1)[0]
    output_path = request_dir / f"{output_basename}.cha"
    CHATFile(doc=doc).write(str(output_path))

    # Provide a download link for the .cha file
    download_url = f"/download/{request_id}/{output_basename}.cha"
    # Read the .cha file contents
    cha_content = ""
    try:
        with open(output_path, "r", encoding="utf-8") as cha_file:
            cha_content = cha_file.read()
    except Exception as e:
        cha_content = f"Error reading .cha file: {e}"

    import html
    cha_content_html = html.escape(cha_content)

    return HTMLResponse(
        f"<h2>Transcription complete!</h2>"
        f"<a href='{download_url}' download>Download .cha file</a>"
        f"<h3>Transcribed .cha file:</h3>"
        f"<pre style='background:#f4f4f4; border:1px solid #ccc; padding:1em; overflow-x:auto;'>{cha_content_html}</pre>"
    )

@app.get("/download/{request_id}/{filename}", include_in_schema=False)
async def download_cha(request_id: str, filename: str):
    from pathlib import Path
    file_path = Path("uploads/tmp") / request_id / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type='application/octet-stream', filename=filename)

@app.get("/api", response_class=HTMLResponse, include_in_schema=False)
async def home():
    return f"""
    <h1><tt>TalkBank | Batchalign2</tt></h1>

    <pre>
    The JSON API welcomes you. 

    If you see this screen, it is likely that the BA2 API is correctly setup.
    Visit <a href="redoc">here</a> for a programming guide/API specifications.

    If you are expecting to *use* Batchalign, you have ended up in the wrong place.
    Feel free to reach out to houjun@cmu.edu / macw@cmu.edu for help.
    </pre>
    """

def run(id:str, command:str, lang:str, name:str):
    workdir = (WORKDIR / id)

    try:
        pipe = BatchalignPipeline.new(command, lang)
        doc = CHATFile(path=workdir/"in.cha").doc
        res = pipe(doc)
        CHATFile(doc=res).write(workdir/"out.cha")

    except Exception:
        exception = traceback.format_exc()
        with open(workdir/"error", 'w') as df:
            df.write(str(exception))

@app.post("/api")
async def submit(
        input: list[UploadFile],
        command: Annotated[str, Form()],
        lang: Annotated[str, Form()],
        background_tasks: BackgroundTasks
):
    """Submit a job for processing."""
    ids = []

    for i in input:
        id = str(uuid.uuid4())
        workdir = (WORKDIR / id)
        workdir.mkdir(exist_ok=True)

        with open(workdir/"in.cha", 'wb') as df:
            df.write((await i.read()))

        name = Path(i.filename).stem

        with open(workdir/"name", 'w') as df:
            df.write(name)

        background_tasks.add_task(run, id=id, command=command, lang=lang, name=name)
        ids.append(id)

    return {"payload": ids, "status": "ok", "key": "submitted"}

@app.get("/api/{id}")
async def status(id):
    """Get status of processed job."""

    id = id.strip()

    if not (WORKDIR / id).is_dir():
        return {"key": "not_found", "status": "error", "message": "The requested job is not found."}

    with open(WORKDIR / id /"name", 'r') as df:
        res = df.read().strip()

    if (WORKDIR / id / "error").is_file():
        with open(str(WORKDIR / id / "error"), 'r') as df:
            return {"key": "job_error", "status": "error", "message": df.read().strip(), "name": res}
    if not (WORKDIR / id / "out.cha").is_file():
        return {"key": "processing", "status": "pending", "message": "The requested job is still processing.", "name": res}

    return {"key": "done", "status": "done", "message": "The requested job is done.", "name": res}

@app.get("/api/get/{id}.cha")
async def get(id):
    """Get processed job."""

    id = id.strip()
    if not (WORKDIR / id).is_dir():
        return HTTPException(status_code=404, detail="Item not found.")
    if (WORKDIR / id / "error").is_file():
        return HTTPException(status_code=400, detail="Item processing errored.")

    return FileResponse(WORKDIR / id / "out.cha", media_type='application/octet-stream')
