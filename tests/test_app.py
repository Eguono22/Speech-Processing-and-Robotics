import os
import tempfile
import unittest
import uuid

from app import app


class RobotApiTests(unittest.TestCase):
    def setUp(self):
        self._old_state_db_path = app.config.get("STATE_DB_PATH")
        self._old_env_file_path = app.config.get("ENV_FILE_PATH")
        self._old_secret_key = app.config.get("SECRET_KEY")
        self._old_env_state_db = os.environ.get("STATE_DB_PATH")
        self._old_env_flask_debug = os.environ.get("FLASK_DEBUG")

        self.db_path = os.path.join(
            tempfile.gettempdir(),
            f"robot_state_{uuid.uuid4().hex}.db",
        )
        self.env_path = os.path.join(
            tempfile.gettempdir(),
            f"robot_env_{uuid.uuid4().hex}.env",
        )
        app.config.update(
            TESTING=True,
            SECRET_KEY="test-secret",
            STATE_DB_PATH=self.db_path,
            ENV_FILE_PATH=self.env_path,
        )
        self.client = app.test_client()

    def tearDown(self):
        app.config["STATE_DB_PATH"] = self._old_state_db_path
        app.config["ENV_FILE_PATH"] = self._old_env_file_path
        app.config["SECRET_KEY"] = self._old_secret_key

        if self._old_env_state_db is None:
            os.environ.pop("STATE_DB_PATH", None)
        else:
            os.environ["STATE_DB_PATH"] = self._old_env_state_db

        if self._old_env_flask_debug is None:
            os.environ.pop("FLASK_DEBUG", None)
        else:
            os.environ["FLASK_DEBUG"] = self._old_env_flask_debug

    def test_state_endpoint_returns_defaults(self):
        response = self.client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["x"], 5)
        self.assertEqual(data["y"], 5)
        self.assertEqual(data["direction"], "north")
        self.assertEqual(data["direction_arrow"], "↑")

    def test_process_moves_robot_and_turns(self):
        # Turn first so movement direction is deterministic and easy to verify.
        response = self.client.post("/api/process", json={"text": "turn right"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["commands"], ["right"])
        self.assertEqual(data["robot_state"]["direction"], "east")

        response = self.client.post("/api/process", json={"text": "go forward"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["commands"], ["forward"])
        self.assertEqual(data["robot_state"]["x"], 6)
        self.assertEqual(data["robot_state"]["y"], 5)
        self.assertEqual(data["direction_arrow"], "→")

    def test_process_rejects_invalid_payload(self):
        response = self.client.post("/api/process", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "No JSON body provided")

        response = self.client.post("/api/process", json={"text": ""})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "No text provided")

    def test_reset_endpoint_restores_defaults(self):
        self.client.post("/api/process", json={"text": "turn right"})
        self.client.post("/api/process", json={"text": "forward"})

        response = self.client.post("/api/reset")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["x"], 5)
        self.assertEqual(data["y"], 5)
        self.assertEqual(data["direction"], "north")
        self.assertEqual(data["direction_arrow"], "↑")

    def test_multi_command_phrase_respects_sentence_order(self):
        response = self.client.post(
            "/api/process",
            json={"text": "turn right then go forward"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["commands"], ["right", "forward"])
        self.assertEqual(data["robot_state"]["x"], 6)
        self.assertEqual(data["robot_state"]["y"], 5)
        self.assertEqual(data["robot_state"]["direction"], "east")

    def test_movement_is_clamped_at_grid_boundaries(self):
        # Move north to top edge.
        for _ in range(20):
            self.client.post("/api/process", json={"text": "forward"})

        response = self.client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["y"], 0)

        # Face west and attempt to move beyond left edge.
        self.client.post("/api/process", json={"text": "turn left"})
        for _ in range(20):
            self.client.post("/api/process", json={"text": "forward"})

        response = self.client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["x"], 0)

    def test_stop_command_is_a_no_op(self):
        before = self.client.get("/api/state").get_json()
        response = self.client.post("/api/process", json={"text": "stop"})
        after = response.get_json()["robot_state"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["commands"], ["stop"])
        self.assertEqual(after["x"], before["x"])
        self.assertEqual(after["y"], before["y"])
        self.assertEqual(after["direction"], before["direction"])

    def test_repeated_commands_are_processed_in_order(self):
        response = self.client.post(
            "/api/process",
            json={"text": "forward forward"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["commands"], ["forward", "forward"])
        self.assertEqual(data["robot_state"]["x"], 5)
        self.assertEqual(data["robot_state"]["y"], 3)

    def test_compound_phrase_is_not_double_counted(self):
        response = self.client.post(
            "/api/process",
            json={"text": "go forward"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["commands"], ["forward"])
        self.assertEqual(data["robot_state"]["x"], 5)
        self.assertEqual(data["robot_state"]["y"], 4)

    def test_settings_get_and_update(self):
        get_response = self.client.get("/api/settings")
        self.assertEqual(get_response.status_code, 200)
        get_data = get_response.get_json()
        self.assertEqual(get_data["state_db_path"], self.db_path)
        self.assertEqual(get_data["effective_state_db_path"], self.db_path)
        self.assertIn("flask_debug", get_data)
        self.assertFalse(get_data["hosted_readonly_mode"])

        new_db = os.path.join(tempfile.gettempdir(), f"override_{uuid.uuid4().hex}.db")
        post_response = self.client.post(
            "/api/settings",
            json={"state_db_path": new_db, "flask_debug": True},
        )
        self.assertEqual(post_response.status_code, 200)
        post_data = post_response.get_json()
        self.assertEqual(post_data["state_db_path"], new_db)
        self.assertEqual(post_data["effective_state_db_path"], new_db)
        self.assertTrue(post_data["flask_debug"])
        self.assertTrue(post_data["restart_required"])

    def test_settings_get_reports_effective_default_db_path(self):
        app.config["STATE_DB_PATH"] = None

        response = self.client.get("/api/settings")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertEqual(data["state_db_path"], "")
        self.assertTrue(data["effective_state_db_path"].endswith("robot_state.db"))

    def test_api_docs_endpoint(self):
        response = self.client.get("/api/docs")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("endpoints", data)
        paths = {endpoint["path"] for endpoint in data["endpoints"]}
        self.assertIn("/api/state", paths)
        self.assertIn("/api/process", paths)
        self.assertIn("/api/settings", paths)
        settings_get = next(
            endpoint
            for endpoint in data["endpoints"]
            if endpoint["path"] == "/api/settings" and endpoint["method"] == "GET"
        )
        settings_post = next(
            endpoint
            for endpoint in data["endpoints"]
            if endpoint["path"] == "/api/settings" and endpoint["method"] == "POST"
        )
        self.assertIn("effective runtime settings", settings_get["description"])
        self.assertIn("active DB path", settings_post["description"])


if __name__ == "__main__":
    unittest.main()
