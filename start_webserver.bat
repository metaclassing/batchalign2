
call .\venv\Scripts\activate.bat

.\venv\Scripts\uvicorn.exe serve:app --port 8083 --host 0.0.0.0
