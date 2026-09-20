# requirements.txt
# Flask>=3.0,<4.0
# datasets>=3.0,<4.0

"""A self-contained emotion-labeling application.

Run locally:
    python app.py

For deployment, set APP_SECRET_KEY and (optionally) ADMIN_PASSWORD in the
environment, then run the WSGI app named ``app`` with a production server.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import hmac
import io
import logging
import os
import secrets
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Callable

from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from jinja2 import DictLoader


DATABASE_PATH = Path(__file__).resolve().with_name("labeling.db")
EMOTIONS = ("anger", "fear", "joy", "love", "sadness", "surprise")
LABEL_LIMIT = 5
PAGE_SIZE = 50
SEED_PER_EMOTION = 10
DATASET_REPOSITORY = "dair-ai/emotion"
DATASET_CONFIG = "split"
DATASET_SPLIT = "train"
DATASET_SOURCE = (
    f"{DATASET_REPOSITORY}:{DATASET_CONFIG}:{DATASET_SPLIT}:"
    f"seed=2026:per_emotion={SEED_PER_EMOTION}"
)

# The default is deliberately supplied for classroom/demo use. Set
# ADMIN_PASSWORD in a deployed environment.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
SECRET_KEY = os.environ.get("APP_SECRET_KEY") or secrets.token_hex(32)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "0") == "1",
)

if "APP_SECRET_KEY" not in os.environ:
    app.logger.warning(
        "APP_SECRET_KEY is unset; sessions will be invalidated after an application restart."
    )


def load_huggingface_seed() -> list[tuple[int, str, str]]:
    """Build a reproducible, balanced seed from the official dataset."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The Hugging Face dataset loader is missing. Install dependencies with "
            "`python -m pip install \\\"Flask>=3.0,<4.0\\\" \\\"datasets>=3.0,<4.0\\\"`."
        ) from error

    dataset = load_dataset(DATASET_REPOSITORY, DATASET_CONFIG, split=DATASET_SPLIT)
    label_names = dataset.features["label"].names
    examples: dict[str, list[str]] = defaultdict(list)

    # A fixed shuffle keeps the same 60 direct-source examples on each clean setup.
    for item in dataset.shuffle(seed=2026):
        emotion = label_names[item["label"]]
        if emotion in EMOTIONS and len(examples[emotion]) < SEED_PER_EMOTION:
            examples[emotion].append(item["text"])
        if all(len(examples[emotion]) == SEED_PER_EMOTION for emotion in EMOTIONS):
            break

    missing = [emotion for emotion in EMOTIONS if len(examples[emotion]) < SEED_PER_EMOTION]
    if missing:
        raise RuntimeError(
            "The Hugging Face dataset did not contain enough examples for: "
            + ", ".join(missing)
        )

    seed_tweets: list[tuple[int, str, str]] = []
    for emotion in EMOTIONS:
        for text in examples[emotion]:
            seed_tweets.append((len(seed_tweets) + 1, text, emotion))
    return seed_tweets


def new_connection() -> sqlite3.Connection:
    """Open an independent connection for the current request/process."""
    connection = sqlite3.connect(DATABASE_PATH, timeout=10, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = new_connection()
    return g.db


@app.teardown_appcontext
def close_db(_: BaseException | None = None) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db() -> None:
    """Create schema and seed empty databases from Hugging Face exactly once."""
    connection = new_connection()
    try:
        with connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tweets (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    true_emotion TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY,
                    participant_id TEXT NOT NULL,
                    tweet_id INTEGER NOT NULL,
                    assigned_label TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY (tweet_id) REFERENCES tweets(id),
                    UNIQUE (participant_id, tweet_id)
                );

                CREATE INDEX IF NOT EXISTS idx_annotations_participant
                    ON annotations(participant_id);
                CREATE INDEX IF NOT EXISTS idx_annotations_label
                    ON annotations(assigned_label);
                CREATE INDEX IF NOT EXISTS idx_annotations_timestamp
                    ON annotations(timestamp DESC);

                CREATE TABLE IF NOT EXISTS app_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

        source_row = connection.execute(
            "SELECT value FROM app_metadata WHERE key = 'tweet_dataset_source'"
        ).fetchone()
        annotation_count = connection.execute(
            "SELECT COUNT(*) AS count FROM annotations"
        ).fetchone()["count"]

        if source_row is not None:
            return
        if annotation_count:
            # Never replace examples that may be referenced by collected labels.
            app.logger.warning(
                "Existing annotated database retained; its tweet source is not changed automatically."
            )
            with connection:
                connection.execute(
                    "INSERT INTO app_metadata (key, value) VALUES (?, ?)",
                    ("tweet_dataset_source", "legacy-local-seed"),
                )
            return

        # Download before acquiring the write transaction, so a slow first download
        # does not hold a SQLite write lock. The datasets library caches this locally.
        seed_tweets = load_huggingface_seed()
        with connection:
            # This handles a pre-Hugging-Face, unannotated database from an earlier
            # version while preserving any database that already has participant data.
            connection.execute("DELETE FROM tweets")
            connection.executemany(
                """
                INSERT INTO tweets (id, text, true_emotion)
                VALUES (?, ?, ?)
                """,
                seed_tweets,
            )
            connection.execute(
                "INSERT INTO app_metadata (key, value) VALUES (?, ?)",
                ("tweet_dataset_source", DATASET_SOURCE),
            )
    finally:
        connection.close()


def clear_labeling_session() -> None:
    for key in ("participant_id", "tweet_ids", "current_index", "completed_participant"):
        session.pop(key, None)


def assignment_from_session() -> tuple[str, list[int], int] | None:
    participant_id = session.get("participant_id")
    tweet_ids = session.get("tweet_ids")
    current_index = session.get("current_index")
    if not isinstance(participant_id, str) or not isinstance(tweet_ids, list):
        return None
    if not isinstance(current_index, int) or not 0 <= current_index < len(tweet_ids):
        return None
    if not tweet_ids or any(not isinstance(tweet_id, int) for tweet_id in tweet_ids):
        return None
    return participant_id, tweet_ids, current_index


def admin_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapped(*args: object, **kwargs: object):
        if not session.get("admin_authenticated"):
            flash("Please sign in to access the administrator dashboard.", "warning")
            return redirect(url_for("admin"))
        return view(*args, **kwargs)

    return wrapped


@app.get("/")
def onboarding():
    return render_template("onboarding.html", title="Emotion Labeling Task")


@app.post("/start")
def start_labeling():
    participant_id = request.form.get("participant_id", "").strip()
    if not participant_id:
        flash("Please enter a participant name or ID to begin.", "error")
        return redirect(url_for("onboarding"))
    if len(participant_id) > 80:
        flash("Participant name or ID must be 80 characters or fewer.", "error")
        return redirect(url_for("onboarding"))

    rows = get_db().execute(
        """
        SELECT id FROM tweets
        WHERE id NOT IN (
            SELECT tweet_id FROM annotations WHERE participant_id = ?
        )
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (participant_id, LABEL_LIMIT),
    ).fetchall()
    tweet_ids = [row["id"] for row in rows]

    if len(tweet_ids) < LABEL_LIMIT:
        flash(
            "This participant ID has fewer than five unlabeled tweets remaining. "
            "Please use a new participant ID.",
            "error",
        )
        return redirect(url_for("onboarding"))

    clear_labeling_session()
    session["participant_id"] = participant_id
    session["tweet_ids"] = tweet_ids
    session["current_index"] = 0
    return redirect(url_for("label"))


@app.route("/label", methods=["GET", "POST"])
def label():
    assignment = assignment_from_session()
    if assignment is None:
        flash("Start a labeling session before submitting annotations.", "warning")
        return redirect(url_for("onboarding"))

    participant_id, tweet_ids, current_index = assignment
    current_tweet_id = tweet_ids[current_index]

    if request.method == "POST":
        submitted_tweet_id = request.form.get("tweet_id", type=int)
        assigned_label = request.form.get("assigned_label", "").lower().strip()
        if submitted_tweet_id != current_tweet_id:
            abort(400, "The submitted tweet does not match the active assignment.")
        if assigned_label not in EMOTIONS:
            flash("Select one of the six emotion labels before continuing.", "error")
            return redirect(url_for("label"))

        # The unique constraint and OR IGNORE make a duplicate browser POST safe.
        with get_db():
            get_db().execute(
                """
                INSERT OR IGNORE INTO annotations
                    (participant_id, tweet_id, assigned_label, timestamp)
                VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                """,
                (participant_id, current_tweet_id, assigned_label),
            )

        next_index = current_index + 1
        if next_index >= len(tweet_ids):
            session["completed_participant"] = participant_id
            session.pop("participant_id", None)
            session.pop("tweet_ids", None)
            session.pop("current_index", None)
            return redirect(url_for("thanks"))

        session["current_index"] = next_index
        return redirect(url_for("label"))

    tweet = get_db().execute(
        "SELECT id, text FROM tweets WHERE id = ?", (current_tweet_id,)
    ).fetchone()
    if tweet is None:
        clear_labeling_session()
        abort(500, "The selected tweet could not be found.")
    return render_template(
        "label.html",
        title=f"Label tweet {current_index + 1}",
        tweet=tweet,
        emotions=EMOTIONS,
        current=current_index + 1,
        total=len(tweet_ids),
        participant_id=participant_id,
    )


@app.get("/thanks")
def thanks():
    participant_id = session.pop("completed_participant", None)
    if not participant_id:
        return redirect(url_for("onboarding"))
    return render_template("thanks.html", title="Labels Saved", participant_id=participant_id)


@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        password = request.form.get("password", "")
        if hmac.compare_digest(password, ADMIN_PASSWORD):
            session["admin_authenticated"] = True
            return redirect(url_for("admin"))
        flash("Incorrect administrator password.", "error")
        return redirect(url_for("admin"))

    if not session.get("admin_authenticated"):
        return render_template("admin_login.html", title="Administrator Sign In")

    page = max(request.args.get("page", 1, type=int), 1)
    offset = (page - 1) * PAGE_SIZE
    db = get_db()
    total_labels = db.execute("SELECT COUNT(*) AS count FROM annotations").fetchone()["count"]
    participant_count = db.execute(
        "SELECT COUNT(DISTINCT participant_id) AS count FROM annotations"
    ).fetchone()["count"]
    counts_by_emotion = {emotion: 0 for emotion in EMOTIONS}
    for row in db.execute(
        "SELECT assigned_label, COUNT(*) AS count FROM annotations GROUP BY assigned_label"
    ):
        counts_by_emotion[row["assigned_label"]] = row["count"]

    annotations = db.execute(
        """
        SELECT a.participant_id, a.tweet_id, t.text AS tweet_text,
               a.assigned_label, a.timestamp
        FROM annotations AS a
        JOIN tweets AS t ON t.id = a.tweet_id
        ORDER BY a.timestamp DESC, a.id DESC
        LIMIT ? OFFSET ?
        """,
        (PAGE_SIZE, offset),
    ).fetchall()
    page_count = max(1, (total_labels + PAGE_SIZE - 1) // PAGE_SIZE)
    if page > page_count and total_labels:
        return redirect(url_for("admin", page=page_count))
    return render_template(
        "admin_dashboard.html",
        title="Administrator Dashboard",
        participant_count=participant_count,
        total_labels=total_labels,
        counts_by_emotion=counts_by_emotion,
        emotions=EMOTIONS,
        annotations=annotations,
        page=page,
        page_count=page_count,
    )


@app.post("/admin/logout")
@admin_required
def admin_logout():
    session.pop("admin_authenticated", None)
    flash("You have been signed out.", "success")
    return redirect(url_for("admin"))


@app.get("/admin/export")
@admin_required
def export_annotations():
    rows = get_db().execute(
        """
        SELECT a.participant_id, a.tweet_id, t.text AS tweet_text,
               a.assigned_label, a.timestamp
        FROM annotations AS a
        JOIN tweets AS t ON t.id = a.tweet_id
        ORDER BY a.id
        """
    ).fetchall()
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["Participant ID", "Tweet ID", "Tweet Text", "Assigned Label", "Timestamp"])
    for row in rows:
        writer.writerow(
            [
                row["participant_id"],
                row["tweet_id"],
                row["tweet_text"],
                row["assigned_label"],
                row["timestamp"],
            ]
        )
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=labeling_data.csv"},
    )


TEMPLATES = {
    "base.html": """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }} · Emotion Labeling</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = { theme: { extend: { colors: { ink: '#172033', mist: '#f5f7fb' } } } }
  </script>
</head>
<body class="min-h-screen bg-mist text-ink antialiased">
  <header class="border-b border-slate-200 bg-white">
    <div class="mx-auto flex max-w-5xl items-center justify-between px-5 py-4">
      <a href="{{ url_for('onboarding') }}" class="flex items-center gap-3 font-semibold tracking-tight text-slate-900">
        <span class="grid h-9 w-9 place-items-center rounded-xl bg-indigo-600 text-lg text-white">✦</span>
        Emotion Labeling
      </a>
      <a href="{{ url_for('admin') }}" class="text-sm font-medium text-slate-500 transition hover:text-indigo-600">Admin</a>
    </div>
  </header>
  <main class="mx-auto max-w-5xl px-5 py-10 sm:py-16">
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% for category, message in messages %}
        <div role="alert" class="mb-6 rounded-xl border px-4 py-3 text-sm {% if category == 'error' %}border-rose-200 bg-rose-50 text-rose-700{% elif category == 'warning' %}border-amber-200 bg-amber-50 text-amber-800{% else %}border-emerald-200 bg-emerald-50 text-emerald-700{% endif %}">{{ message }}</div>
      {% endfor %}
    {% endwith %}
    {% block content %}{% endblock %}
  </main>
  <footer class="pb-8 text-center text-xs text-slate-400">Emotion Data Labeling Task</footer>
</body>
</html>
""",
    "onboarding.html": """
{% extends 'base.html' %}
{% block content %}
<section class="mx-auto max-w-2xl">
  <div class="mb-8 text-center">
    <p class="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-indigo-600">NLP research task</p>
    <h1 class="text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">Label the feeling behind each tweet.</h1>
    <p class="mx-auto mt-5 max-w-xl text-base leading-7 text-slate-600">You will read five short Twitter-style messages and choose the single emotion that best represents each one. Your responses are saved as you proceed.</p>
  </div>
  <div class="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
    <h2 class="text-lg font-semibold text-slate-900">Before you begin</h2>
    <ol class="mt-4 space-y-3 text-sm leading-6 text-slate-600">
      <li class="flex gap-3"><span class="font-semibold text-indigo-600">01</span><span>Read each message carefully and select its primary emotional tone.</span></li>
      <li class="flex gap-3"><span class="font-semibold text-indigo-600">02</span><span>Choose exactly one label: anger, fear, joy, love, sadness, or surprise.</span></li>
      <li class="flex gap-3"><span class="font-semibold text-indigo-600">03</span><span>Submit all five labels. There are no right or wrong answers—use your best judgment.</span></li>
    </ol>
    <form method="post" action="{{ url_for('start_labeling') }}" class="mt-8 border-t border-slate-100 pt-6">
      <label for="participant_id" class="mb-2 block text-sm font-medium text-slate-700">Participant name / ID</label>
      <div class="flex flex-col gap-3 sm:flex-row">
        <input id="participant_id" name="participant_id" type="text" maxlength="80" required autocomplete="name" placeholder="e.g., participant-001" class="min-w-0 flex-1 rounded-xl border border-slate-300 px-4 py-3 text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100">
        <button type="submit" class="rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-4 focus:ring-indigo-200">Start labeling <span aria-hidden="true">→</span></button>
      </div>
      <p class="mt-3 text-xs text-slate-400">Use a unique ID. Existing IDs resume only with previously unseen messages.</p>
    </form>
  </div>
</section>
{% endblock %}
""",
    "label.html": """
{% extends 'base.html' %}
{% block content %}
<section class="mx-auto max-w-3xl">
  <div class="mb-8 flex items-end justify-between gap-4">
    <div><p class="text-sm font-semibold uppercase tracking-[0.18em] text-indigo-600">Participant: {{ participant_id }}</p><h1 class="mt-2 text-3xl font-bold tracking-tight text-slate-900">What emotion is expressed?</h1></div>
    <p class="shrink-0 text-sm font-medium text-slate-500">Tweet {{ current }} of {{ total }}</p>
  </div>
  <div class="mb-8 h-2 overflow-hidden rounded-full bg-slate-200" role="progressbar" aria-valuenow="{{ current }}" aria-valuemin="1" aria-valuemax="{{ total }}">
    <div class="h-full rounded-full bg-indigo-600 transition-all" style="width: {{ (current / total * 100)|round }}%"></div>
  </div>
  <form method="post" action="{{ url_for('label') }}" class="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-9">
    <input type="hidden" name="tweet_id" value="{{ tweet.id }}">
    <blockquote class="rounded-xl border-l-4 border-indigo-500 bg-indigo-50/70 px-5 py-6 text-xl font-medium leading-8 text-slate-800 sm:text-2xl">“{{ tweet.text }}”</blockquote>
    <fieldset class="mt-8">
      <legend class="mb-4 text-sm font-semibold text-slate-700">Select the primary emotion</legend>
      <div class="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {% for emotion in emotions %}
        <label class="cursor-pointer">
          <input class="peer sr-only" type="radio" name="assigned_label" value="{{ emotion }}" {% if loop.first %}required{% endif %}>
          <span class="flex items-center justify-center rounded-xl border border-slate-200 bg-white px-4 py-4 text-sm font-semibold capitalize text-slate-700 transition hover:border-indigo-300 hover:bg-indigo-50 peer-checked:border-indigo-600 peer-checked:bg-indigo-600 peer-checked:text-white peer-focus-visible:ring-4 peer-focus-visible:ring-indigo-200">{{ emotion }}</span>
        </label>
        {% endfor %}
      </div>
    </fieldset>
    <div class="mt-8 flex items-center justify-between border-t border-slate-100 pt-6">
      <p class="text-xs text-slate-400">Your label is recorded when you continue.</p>
      <button type="submit" class="rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-4 focus:ring-indigo-200">Submit &amp; Next <span aria-hidden="true">→</span></button>
    </div>
  </form>
</section>
{% endblock %}
""",
    "thanks.html": """
{% extends 'base.html' %}
{% block content %}
<section class="mx-auto max-w-xl text-center">
  <div class="rounded-2xl border border-emerald-100 bg-white p-8 shadow-sm sm:p-12">
    <div class="mx-auto grid h-16 w-16 place-items-center rounded-full bg-emerald-100 text-3xl text-emerald-600">✓</div>
    <p class="mt-6 text-sm font-semibold uppercase tracking-[0.18em] text-emerald-600">Submission complete</p>
    <h1 class="mt-3 text-3xl font-bold tracking-tight text-slate-900">Thank you, {{ participant_id }}!</h1>
    <p class="mt-4 leading-7 text-slate-600">Your five emotion labels were successfully stored. Your contribution helps improve emotion understanding in language data.</p>
    <a href="{{ url_for('onboarding') }}" class="mt-8 inline-flex rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-4 focus:ring-indigo-200">Start a new participant session</a>
  </div>
</section>
{% endblock %}
""",
    "admin_login.html": """
{% extends 'base.html' %}
{% block content %}
<section class="mx-auto max-w-md">
  <div class="rounded-2xl border border-slate-200 bg-white p-7 shadow-sm sm:p-9">
    <p class="text-sm font-semibold uppercase tracking-[0.18em] text-indigo-600">Restricted area</p>
    <h1 class="mt-2 text-3xl font-bold tracking-tight text-slate-900">Administrator sign in</h1>
    <p class="mt-3 text-sm leading-6 text-slate-500">Enter the dashboard password to view annotations and export data.</p>
    <form method="post" action="{{ url_for('admin') }}" class="mt-7">
      <label for="password" class="mb-2 block text-sm font-medium text-slate-700">Password</label>
      <input id="password" name="password" type="password" required autocomplete="current-password" class="w-full rounded-xl border border-slate-300 px-4 py-3 outline-none transition focus:border-indigo-500 focus:ring-4 focus:ring-indigo-100">
      <button type="submit" class="mt-5 w-full rounded-xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-4 focus:ring-indigo-200">Sign in</button>
    </form>
  </div>
</section>
{% endblock %}
""",
    "admin_dashboard.html": """
{% extends 'base.html' %}
{% block content %}
<section>
  <div class="mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
    <div><p class="text-sm font-semibold uppercase tracking-[0.18em] text-indigo-600">Administration</p><h1 class="mt-2 text-3xl font-bold tracking-tight text-slate-900">Labeling dashboard</h1></div>
    <div class="flex gap-3"><a href="{{ url_for('export_annotations') }}" class="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700">Download CSV</a><form method="post" action="{{ url_for('admin_logout') }}"><button type="submit" class="rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50">Sign out</button></form></div>
  </div>
  <div class="grid gap-4 sm:grid-cols-2">
    <div class="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p class="text-sm font-medium text-slate-500">Active participants</p><p class="mt-2 text-3xl font-bold text-slate-900">{{ participant_count }}</p></div>
    <div class="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p class="text-sm font-medium text-slate-500">Labels logged</p><p class="mt-2 text-3xl font-bold text-slate-900">{{ total_labels }}</p></div>
  </div>
  <div class="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
    {% for emotion in emotions %}<div class="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"><p class="text-xs font-semibold capitalize text-slate-500">{{ emotion }}</p><p class="mt-1 text-2xl font-bold text-slate-900">{{ counts_by_emotion[emotion] }}</p></div>{% endfor %}
  </div>
  <div class="mt-8 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
    <div class="border-b border-slate-100 px-5 py-4"><h2 class="font-semibold text-slate-900">Stored annotations</h2></div>
    <div class="overflow-x-auto"><table class="min-w-full text-left text-sm"><thead class="bg-slate-50 text-xs uppercase tracking-wide text-slate-500"><tr><th class="px-5 py-3 font-semibold">Participant ID</th><th class="px-5 py-3 font-semibold">Tweet ID</th><th class="min-w-96 px-5 py-3 font-semibold">Tweet Text</th><th class="px-5 py-3 font-semibold">Assigned Label</th><th class="whitespace-nowrap px-5 py-3 font-semibold">Timestamp</th></tr></thead><tbody class="divide-y divide-slate-100 text-slate-700">
      {% for annotation in annotations %}<tr><td class="whitespace-nowrap px-5 py-4 font-medium">{{ annotation.participant_id }}</td><td class="px-5 py-4">{{ annotation.tweet_id }}</td><td class="px-5 py-4 leading-6">{{ annotation.tweet_text }}</td><td class="px-5 py-4"><span class="rounded-full bg-indigo-50 px-2.5 py-1 text-xs font-semibold capitalize text-indigo-700">{{ annotation.assigned_label }}</span></td><td class="whitespace-nowrap px-5 py-4 text-xs text-slate-500">{{ annotation.timestamp }}</td></tr>{% else %}<tr><td colspan="5" class="px-5 py-10 text-center text-slate-500">No labels have been submitted yet.</td></tr>{% endfor %}
    </tbody></table></div>
    {% if page_count > 1 %}<nav class="flex items-center justify-between border-t border-slate-100 px-5 py-4 text-sm"><span class="text-slate-500">Page {{ page }} of {{ page_count }}</span><div class="flex gap-2">{% if page > 1 %}<a class="rounded-lg border border-slate-300 px-3 py-1.5 hover:bg-slate-50" href="{{ url_for('admin', page=page - 1) }}">Previous</a>{% endif %}{% if page < page_count %}<a class="rounded-lg border border-slate-300 px-3 py-1.5 hover:bg-slate-50" href="{{ url_for('admin', page=page + 1) }}">Next</a>{% endif %}</div></nav>{% endif %}
  </div>
</section>
{% endblock %}
""",
}

app.jinja_loader = DictLoader(TEMPLATES)


with app.app_context():
    init_db()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1", threaded=True)
