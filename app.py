import os
import re
import sqlite3
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, session


def _get_env_path() -> Path:
    return Path(app.config.get("ENV_FILE_PATH", ".env"))


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return
    for key, value in _read_env_file(path).items():
        os.environ.setdefault(key, value)


_load_dotenv()

app = Flask(__name__)
# Use SECRET_KEY env var in production; default remains stable so session
# cookies continue working across development restarts.
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
app.config["STATE_DB_PATH"] = os.environ.get("STATE_DB_PATH")
app.config["ENV_FILE_PATH"] = os.environ.get("ENV_FILE_PATH", ".env")

GRID_SIZE = 10
DEFAULT_X = 5
DEFAULT_Y = 5
DEFAULT_DIRECTION = "north"

# Spatial command keyword mappings
SPATIAL_COMMANDS = {
    "forward": ["go forward", "move forward", "forward", "ahead", "advance", "march"],
    "backward": ["go backward", "move backward", "backward", "back", "reverse", "retreat", "behind"],
    "left": ["left", "turn left"],
    "right": ["right", "turn right"],
    "stop": ["stop", "halt", "freeze", "pause", "wait"],
    "reset": ["reset", "restart", "home", "origin", "start"],
}

# Direction turn orders
TURN_LEFT = ["north", "west", "south", "east"]
TURN_RIGHT = ["north", "east", "south", "west"]

# Direction labels for compass arrows
DIRECTION_ARROWS = {
    "north": "↑",
    "south": "↓",
    "east": "→",
    "west": "←",
}

def _get_db_path() -> str:
    configured = app.config.get("STATE_DB_PATH")
    if configured:
        return configured
    os.makedirs(app.instance_path, exist_ok=True)
    return os.path.join(app.instance_path, "robot_state.db")


def _init_db() -> None:
    with sqlite3.connect(_get_db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS robot_states (
                sid TEXT PRIMARY KEY,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                direction TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _make_default_state() -> dict:
    return {
        "x": DEFAULT_X,
        "y": DEFAULT_Y,
        "direction": DEFAULT_DIRECTION,
        "grid_size": GRID_SIZE,
    }


def _load_state(sid: str) -> dict | None:
    _init_db()
    with sqlite3.connect(_get_db_path()) as conn:
        row = conn.execute(
            "SELECT x, y, direction FROM robot_states WHERE sid = ?",
            (sid,),
        ).fetchone()
    if row is None:
        return None
    return {
        "x": row[0],
        "y": row[1],
        "direction": row[2],
        "grid_size": GRID_SIZE,
    }


def _save_state(sid: str, state: dict) -> None:
    _init_db()
    with sqlite3.connect(_get_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO robot_states (sid, x, y, direction)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sid) DO UPDATE SET
                x = excluded.x,
                y = excluded.y,
                direction = excluded.direction
            """,
            (sid, state["x"], state["y"], state["direction"]),
        )
        conn.commit()


def _get_state() -> dict:
    sid = session.get("sid")
    if sid is None:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    state = _load_state(sid)
    if state is None:
        state = _make_default_state()
        _save_state(sid, state)
    return state


def _apply_command(state: dict, command: str) -> None:
    gs = state["grid_size"]
    direction = state["direction"]

    if command == "forward":
        if direction == "north":
            state["y"] = max(0, state["y"] - 1)
        elif direction == "south":
            state["y"] = min(gs - 1, state["y"] + 1)
        elif direction == "east":
            state["x"] = min(gs - 1, state["x"] + 1)
        elif direction == "west":
            state["x"] = max(0, state["x"] - 1)
    elif command == "backward":
        if direction == "north":
            state["y"] = min(gs - 1, state["y"] + 1)
        elif direction == "south":
            state["y"] = max(0, state["y"] - 1)
        elif direction == "east":
            state["x"] = max(0, state["x"] - 1)
        elif direction == "west":
            state["x"] = min(gs - 1, state["x"] + 1)
    elif command == "left":
        idx = TURN_LEFT.index(direction)
        state["direction"] = TURN_LEFT[(idx + 1) % len(TURN_LEFT)]
    elif command == "right":
        idx = TURN_RIGHT.index(direction)
        state["direction"] = TURN_RIGHT[(idx + 1) % len(TURN_RIGHT)]
    elif command == "reset":
        state["x"] = DEFAULT_X
        state["y"] = DEFAULT_Y
        state["direction"] = DEFAULT_DIRECTION
    elif command == "stop":
        # "stop" is intentionally a no-op for localization state.
        return


def _recognize_spatial_context(text: str) -> list[str]:
    """Return a list of spatial commands detected in *text* (in order)."""
    text_lower = text.lower()
    matches: list[tuple[int, int, str]] = []

    for command, keywords in SPATIAL_COMMANDS.items():
        for keyword in keywords:
            pattern = r"\b" + r"\s+".join(re.escape(part) for part in keyword.split()) + r"\b"
            for match in re.finditer(pattern, text_lower):
                matches.append((match.start(), match.end(), command))

    # Prefer longer matches when starting at the same index
    # (e.g. "turn right" over "right"), then keep only non-overlapping
    # spans to avoid duplicate detections for one spoken phrase.
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[str] = []
    last_end = -1
    for start, end, command in matches:
        if start < last_end:
            continue
        selected.append(command)
        last_end = end
    return selected


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_speech():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    commands = _recognize_spatial_context(text)
    state = _get_state()
    for command in commands:
        _apply_command(state, command)
    _save_state(session["sid"], state)

    return jsonify(
        {
            "text": text,
            "commands": commands,
            "robot_state": state.copy(),
            "direction_arrow": DIRECTION_ARROWS[state["direction"]],
        }
    )


@app.route("/api/state", methods=["GET"])
def get_state():
    state = _get_state()
    return jsonify(
        {
            **state,
            "direction_arrow": DIRECTION_ARROWS[state["direction"]],
        }
    )


@app.route("/api/reset", methods=["POST"])
def reset_robot():
    state = _get_state()
    state.update(
        {
            "x": DEFAULT_X,
            "y": DEFAULT_Y,
            "direction": DEFAULT_DIRECTION,
        }
    )
    _save_state(session["sid"], state)
    return jsonify(
        {
            **state,
            "direction_arrow": DIRECTION_ARROWS[state["direction"]],
        }
    )


@app.route("/api/settings", methods=["GET"])
def get_settings():
    debug_enabled = os.environ.get("FLASK_DEBUG", "0") == "1"
    return jsonify(
        {
            "state_db_path": app.config.get("STATE_DB_PATH") or "",
            "flask_debug": debug_enabled,
            "env_file_path": str(_get_env_path()),
        }
    )


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json(silent=True) or {}
    state_db_path = str(data.get("state_db_path", "")).strip()
    flask_debug = bool(data.get("flask_debug", False))

    env_path = _get_env_path()
    env_values = _read_env_file(env_path)
    env_values["FLASK_DEBUG"] = "1" if flask_debug else "0"

    if state_db_path:
        env_values["STATE_DB_PATH"] = state_db_path
        app.config["STATE_DB_PATH"] = state_db_path
    else:
        env_values.pop("STATE_DB_PATH", None)
        app.config["STATE_DB_PATH"] = None

    _write_env_file(env_path, env_values)
    os.environ["FLASK_DEBUG"] = env_values["FLASK_DEBUG"]
    if "STATE_DB_PATH" in env_values:
        os.environ["STATE_DB_PATH"] = env_values["STATE_DB_PATH"]
    else:
        os.environ.pop("STATE_DB_PATH", None)

    return jsonify(
        {
            "state_db_path": app.config.get("STATE_DB_PATH") or "",
            "flask_debug": flask_debug,
            "restart_required": True,
        }
    )


@app.route("/api/docs", methods=["GET"])
def api_docs():
    return jsonify(
        {
            "endpoints": [
                {
                    "method": "GET",
                    "path": "/api/state",
                    "description": "Get the current robot state for this session.",
                },
                {
                    "method": "POST",
                    "path": "/api/process",
                    "description": "Process speech/text command and update robot state.",
                    "json_body": {"text": "turn right then forward"},
                },
                {
                    "method": "POST",
                    "path": "/api/reset",
                    "description": "Reset robot to default position and direction.",
                },
                {
                    "method": "GET",
                    "path": "/api/settings",
                    "description": "Read effective runtime settings.",
                },
                {
                    "method": "POST",
                    "path": "/api/settings",
                    "description": "Save settings to .env for next restart.",
                    "json_body": {
                        "state_db_path": "instance/robot_state.db",
                        "flask_debug": False,
                    },
                },
            ]
        }
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"[startup] Robot state DB: {_get_db_path()}")
    app.run(debug=debug)
