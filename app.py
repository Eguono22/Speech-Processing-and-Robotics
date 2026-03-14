import os
import uuid

from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
# Use SECRET_KEY env var in production; fall back to a random key for development.
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)

GRID_SIZE = 10
DEFAULT_X = 5
DEFAULT_Y = 5
DEFAULT_DIRECTION = "north"

# Spatial command keyword mappings
SPATIAL_COMMANDS = {
    "forward": ["forward", "ahead", "go", "advance", "march", "move"],
    "backward": ["backward", "back", "reverse", "retreat", "behind"],
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

# In-memory per-session robot states.
# NOTE: state is lost on server restart and is not shared across
# multiple worker processes. For production use, replace with a
# persistent store (e.g. Redis or a database).
_robot_states: dict[str, dict] = {}


def _make_default_state() -> dict:
    return {
        "x": DEFAULT_X,
        "y": DEFAULT_Y,
        "direction": DEFAULT_DIRECTION,
        "grid_size": GRID_SIZE,
    }


def _get_state() -> dict:
    sid = session.get("sid")
    if sid is None or sid not in _robot_states:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _robot_states[sid] = _make_default_state()
    return _robot_states[sid]


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


def _recognize_spatial_context(text: str) -> list[str]:
    """Return a list of spatial commands detected in *text* (in order)."""
    text_lower = text.lower()
    recognized: list[str] = []
    for command, keywords in SPATIAL_COMMANDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                recognized.append(command)
                break
    return recognized


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
    return jsonify(
        {
            **state,
            "direction_arrow": DIRECTION_ARROWS[state["direction"]],
        }
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)
