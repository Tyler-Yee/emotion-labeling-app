# Emotion Data Labeling

A Flask web application for collecting emotion labels for Twitter-style text. It presents five randomized messages per participant and stores labels in SQLite. The first initialization downloads a reproducible, balanced 60-message sample from [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion).

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. The default administrator password is `admin123`; set `ADMIN_PASSWORD` before deployment.

## Deploy on Render

This repository includes `render.yaml`, which is the recommended Render configuration. It creates a Python web service, installs the packages in `requirements.txt`, starts the app with Gunicorn, and attaches a persistent disk for SQLite.

1. Commit and push these files to GitHub.
2. In Render, select **New > Blueprint** and connect the repository.
3. Review the proposed `emotion-labeling` service. Set a strong value for `ADMIN_PASSWORD` when Render requests it.
4. Create the Blueprint and wait for the first deploy to finish.
5. Open the generated `https://<service-name>.onrender.com` URL and share it with participants.

Render automatically generates `APP_SECRET_KEY`. The `/healthz` endpoint is configured as the service health check.

### Important persistence note

Render's free web services use an ephemeral filesystem, so a local SQLite database would be erased on restarts and deployments. The included Blueprint uses a 1 GB persistent disk mounted at `/var/data` and requires a paid Render web-service plan. Keep this service at one instance: a persistent disk can only attach to one Render instance, and SQLite is designed here for that single-instance use case.

The first startup downloads the official dataset from Hugging Face. The selected sample and all subsequent annotations are retained on the disk, so later restarts do not need to download or reseed it.

`Procfile` is included for compatibility with platforms that use it, but Render uses `render.yaml` as the source of truth.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `APP_SECRET_KEY` | Required in deployment; signs Flask sessions. Render generates it from the Blueprint. |
| `ADMIN_PASSWORD` | Dashboard password. Set a strong secret in Render. |
| `LABELING_DATABASE_PATH` | SQLite file location. Render uses `/var/data/labeling.db`. |
| `SESSION_COOKIE_SECURE` | Set to `1` in production to send session cookies only over HTTPS. |
| `PORT` | Supplied automatically by Render. |

## Manual Render setup

If you do not use the Blueprint, create a **Web Service** with these settings:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --workers 1 --threads 4 --bind 0.0.0.0:$PORT app:app`
- Health check path: `/healthz`
- Disk mount path: `/var/data` (1 GB or larger)
- `LABELING_DATABASE_PATH`: `/var/data/labeling.db`
- `SESSION_COOKIE_SECURE`: `1`
- Set strong values for `APP_SECRET_KEY` and `ADMIN_PASSWORD`
