# Emotion Data Labeling

A Flask web application for collecting emotion labels for Twitter-style text. It presents five randomized messages per participant and stores labels in SQLite locally or PostgreSQL when `DATABASE_URL` is configured. The first initialization downloads a reproducible, balanced 60-message sample from [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion).

The application pins Render to Python 3.12 through `.python-version`. This avoids compatibility problems with Python 3.14 and the Hugging Face dataset loader.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. The default administrator password is `admin123`; set `ADMIN_PASSWORD` before deployment.

## Deploy on Render

This repository includes a no-card `render.yaml` Blueprint. It creates a Free Python web service and a Free Render Postgres database, connects them through Render's private network, installs the packages in `requirements.txt`, and starts the app with Gunicorn.

1. Commit and push these files to GitHub.
2. In Render, select **New > Blueprint** and connect the repository.
3. Review the proposed `emotion-labeling` service. Set a strong value for `ADMIN_PASSWORD` when Render requests it.
4. Create the Blueprint and wait for the first deploy to finish.
5. Open the generated `https://<service-name>.onrender.com` URL and share it with participants.

Render automatically generates `APP_SECRET_KEY`. The `/healthz` endpoint is configured as the service health check.

### Free-tier limitations

Render's Free web service spins down after 15 minutes without traffic, so the first request after inactivity can take about a minute. This does **not** erase labels: labels are stored in Render Postgres, not the web service filesystem.

Free Render Postgres has a 1 GB limit and expires after 30 days. Before it expires, export your labels from `/admin/export`; upgrade the database to retain it longer if the study needs more time. Render allows one active Free Postgres database per workspace.

The first web-service startup downloads the official dataset from Hugging Face and stores the selected sample in Postgres. Later restarts do not need to download or reseed it.

`Procfile` is included for compatibility with platforms that use it, but Render uses `render.yaml` as the source of truth.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `APP_SECRET_KEY` | Required in deployment; signs Flask sessions. Render generates it from the Blueprint. |
| `ADMIN_PASSWORD` | Dashboard password. Set a strong secret in Render. |
| `DATABASE_URL` | PostgreSQL connection URL. Render injects it automatically. When absent, the app uses local SQLite. |
| `LABELING_DATABASE_PATH` | Optional local SQLite file location. Defaults to `labeling.db` next to `app.py`. |
| `SESSION_COOKIE_SECURE` | Set to `1` in production to send session cookies only over HTTPS. |
| `PORT` | Supplied automatically by Render. |

## Manual Render setup

If you do not use the Blueprint, create a **Web Service** with these settings:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --workers 1 --threads 4 --bind 0.0.0.0:$PORT app:app`
- Health check path: `/healthz`
- Create a Free Render Postgres database in the same region
- `DATABASE_URL`: use the database's internal connection string
- `SESSION_COOKIE_SECURE`: `1`
- Set strong values for `APP_SECRET_KEY` and `ADMIN_PASSWORD`
