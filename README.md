# MeetingMind

MeetingMind turns a meeting transcript into a summary, decisions, action items
(owner, deadline, and priority), risks, and open questions. Review the results
before sharing or importing them into another system.

## How it works

Transcript -> Claude API -> Pydantic validation (with one repair retry) ->
Streamlit results -> Markdown, CSV, or Jira-importable CSV.

## How a transcript becomes downloadable notes

1. The browser accepts a UTF-8 `.txt` file, or you can paste text or load the
   sample meeting. The app reads the uploaded file into text and keeps it in
   the current Streamlit session so it can display it in the transcript box.
   The app code does not write the uploaded transcript to a project file.
2. Nothing is sent to Claude until you click **Extract notes**. At that point,
   the app server sends the transcript to Anthropic's Claude API using the key
   in `.env` (or Streamlit secrets when hosted). Use only transcripts you are
   allowed to share with that service; real API requests can incur charges.
3. `extractor.py` asks Claude for a JSON response, checks it with Pydantic, and
   retries once if the response is invalid. The validated notes are kept for
   the current app session and displayed in the browser.
4. The download buttons format those notes as Markdown or CSV in memory and
   send the file to your browser to save. Jira CSV is only an import file; it
   does not connect to Jira or create tickets. The app does not permanently
   save the transcript or notes to the project folder.

## What the project files do

- `app.py` builds the Streamlit page, accepts pasted or uploaded transcripts,
  calls the extractor when you click **Extract notes**, displays the result,
  and provides download buttons.
- `extractor.py` defines the expected note fields, sends the transcript to
  Claude, validates the response, and formats the downloadable files.
- `evaluate.py` calls Claude on the labelled sample transcripts and prints
  approximate quality scores. Running it makes real API requests.
- `sample_data/` contains synthetic transcripts for trying the app and JSON
  labels used by the evaluation.
- `tests/test_extractor.py` checks extraction, validation, retries, and exports
  with a fake Claude client, so the unit tests do not call the paid API.
- `.env` holds your private local API key; `.env.example` is a safe template.
  Never upload `.env` or put the key in source code.
- `requirements.txt` lists the Python packages the project needs.

The functions in the Python files have docstrings that describe their purpose.
For example, `extract_notes()` handles the Claude request and validation,
`to_jira_csv()` formats Jira import data, and `get_api_key()` reads the key
without displaying it.

## Requirements

- Python 3.10 or newer
- An Anthropic API key with API billing enabled

## Run locally on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Add your active API key to `.env` if it is not already configured:

```text
ANTHROPIC_API_KEY=your-real-key
```

Then start the app:

```powershell
streamlit run app.py
```

## Run and use the app in your browser

1. Open the project folder in VS Code, then open **Terminal > New Terminal**.
2. In PowerShell, go to the project folder and activate its virtual environment:

   ```powershell
   Set-Location "C:\Users\ADMIN\Documents\Deepthi Project"
   .\.venv\Scripts\Activate.ps1
   ```

   If this is a fresh setup, follow the installation steps above first. Make
   sure `.env` contains your active `ANTHROPIC_API_KEY`; never paste the key
   into the terminal command, README, or browser.
3. Start the web app:

   ```powershell
   streamlit run app.py
   ```

   Keep this terminal open while using the app. Streamlit will show a local URL.
   Open **http://localhost:8501** in your browser (or Ctrl+click the URL in the
   terminal).
4. In MeetingMind, click **Load sample meeting**, or paste a transcript into
   the text area or upload a UTF-8 `.txt` file.
5. Click **Extract notes** and wait for the results. This sends the transcript
   to Claude through the API and may incur API charges.
6. Review the summary, decisions, action items, risks, and open questions.
   Check the results before sharing them.
7. Use **Download Markdown**, **Download CSV**, or **Download Jira CSV** to
   save the output.
8. To stop the app, return to the terminal and press **Ctrl+C**.

The default model is configurable with `MEETINGMIND_MODEL` or in the app
sidebar. If Claude reports that the default model is unavailable, set that
variable to a model enabled for your Anthropic account.

## Tests and quality evaluation

```powershell
python -m pytest -q
python evaluate.py
```

Unit tests use a fake Claude client and do not make API calls. `evaluate.py`
uses the real API and the synthetic, hand-labelled transcripts in `sample_data`.
Its keyword-based scores are approximate; do not report metrics as measured
results until you have run the evaluation and reviewed its output.

## Dependencies

See [requirements.txt](requirements.txt). Never commit `.env` or use confidential
meeting transcripts in a public demo.
