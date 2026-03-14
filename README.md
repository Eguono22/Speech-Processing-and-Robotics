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
| 🔄 Per-session state | Each browser tab keeps its own robot state via Flask sessions |

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

# 4. Run the development server
python app.py
```

Open your browser at **http://127.0.0.1:5000**.

> **Note:** The Web Speech API requires a secure context (HTTPS) or localhost. It is supported in Chrome/Edge and partially in other browsers.

---

## Project Structure

```
Speech-Processing-and-Robotics/
├── app.py               # Flask application & spatial-context logic
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
