# Speech-Processing-and-Robotics

**Design of a Simultaneous Mobile Robot Localization and Spatial Context Recognition System**

A Flask web application that lets you drive a simulated robot on a 2-D grid using **voice commands** (via the browser's Web Speech API) or typed text. The backend recognises spatial commands and updates the robot's position and orientation in real time.

---

## Features

| Feature | Detail |
|---|---|
| 🎙️ Speech input | Uses the browser Web Speech API for hands-free control |
| ⌨️ Text input | Fallback text box for any browser |
| 🗺️ Live grid map | Canvas-rendered 10 × 10 grid with robot position, direction arrow, and movement trail |
| 🏷️ Spatial context | Highlights the recognised spatial commands (forward, backward, turn left/right, stop, reset) |
| 📋 Command log | Timestamped history of every command sent |
| 💾 Persistent state | Per-session robot state is stored in SQLite and survives server restarts |

---

## Supported Commands

Say or type any sentence containing these keywords:

| Command | Example phrases |
|---|---|
| `forward` | "go forward", "move ahead", "advance" |
| `backward` | "go backward", "reverse", "retreat" |
| `left` | "turn left", "left" |
| `right` | "turn right", "right" |
| `stop` | "stop", "halt", "freeze" |
| `reset` | "reset", "home", "restart" |

---

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Eguono22/Speech-Processing-and-Robotics.git
cd Speech-Processing-and-Robotics

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) set environment variables
cp .env.example .env    # Windows PowerShell: Copy-Item .env.example .env

# 5. Run the development server
python app.py
```

Open your browser at **http://127.0.0.1:5000**.

> **Note:** The Web Speech API requires a secure context (HTTPS) or localhost. It is supported in Chrome/Edge and partially in other browsers.

---

## Run Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

CI runs the same test command on every push and pull request via
`.github/workflows/ci.yml`.

---

## Docker

```bash
# Build image
docker build -t speech-robot-localization .

# Run container
docker run --rm -p 5000:5000 speech-robot-localization
```

Open `http://127.0.0.1:5000`.

---

## Deploy As a Web App

This project is a Flask web application and can be deployed on serverless
platforms (for example Vercel) or traditional WSGI hosts.

- Entry point for serverless: `api/index.py`
- WSGI entry point: `wsgi.py` (`application`)
- Cloud runtime port: set automatically via `PORT`

### Notes for Hosted Environments

- In hosted/serverless mode, runtime writes to `.env` are disabled.
- Use platform environment variables (for example `SECRET_KEY`,
  `STATE_DB_PATH`, `FLASK_DEBUG`) in your deployment settings instead.

---

## API Reference

### `GET /api/state`

Returns current robot state for the session.

### `POST /api/process`

Request body:

```json
{
  "text": "turn right then forward"
}
```

Response includes `commands`, `robot_state`, and `direction_arrow`.

### `POST /api/reset`

Resets robot to default position and direction.

### `GET /api/settings`

Returns effective settings shown in the UI settings card.

### `POST /api/settings`

Request body:

```json
{
  "state_db_path": "instance/robot_state.db",
  "flask_debug": false
}
```

Saves settings to `.env`. Restart the server to apply debug-mode changes.

### `GET /api/docs`

Returns machine-readable endpoint documentation as JSON.

---

## Persistence Notes

- Robot state is persisted in SQLite at `instance/robot_state.db` by default.
- Override the path with `STATE_DB_PATH` in `.env` if needed.
- Set `SECRET_KEY` in `.env` for secure and stable session handling.

---

## Project Structure

```
Speech-Processing-and-Robotics/
├── app.py               # Flask application & spatial-context logic
├── tests/
│   └── test_app.py      # API and robot-state unit tests
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Main UI template
└── static/
    ├── css/style.css    # Styles
    └── js/main.js       # Canvas renderer, Web Speech API, API calls
```

---

## License

MIT © 2026 Eguonorghene Emmanuel Adomi
