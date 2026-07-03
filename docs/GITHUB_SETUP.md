# GitHub setup

This project is intended to keep source code, scripts, and operating notes in GitHub while keeping race data, model artifacts, odds snapshots, outputs, and local tools out of Git.

## What is tracked

- `config/`
- `docs/*.md`
- `prompts/`
- `scripts/`
- `src/`
- `.gitignore`

## What is intentionally not tracked

- `data/`
- `date/raw/`, `date/interim/`, `date/processed/`
- `models/`
- `outputs/`
- `tools/`
- `.env`
- local PDFs/XLSX reference manuals

## One-time GitHub steps

1. Create an empty **private** GitHub repository.
2. Do not initialize it with README, `.gitignore`, or license.
3. Copy the repository URL.
4. Add the remote locally:

```powershell
.\tools\git\mingit\cmd\git.exe remote add origin https://github.com/<owner>/<repo>.git
.\tools\git\mingit\cmd\git.exe push -u origin main
```

## Safety rules

- Do not commit raw TARGET/JRA-VAN exports.
- Do not commit live odds, model files, or dashboard outputs.
- Do not commit LINE tokens or other credentials.
- Treat GitHub as source backup and collaboration history, not as the data store.
