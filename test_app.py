import unittest
import app as app_module


class FakeClassifier:
    """
    A fake classifier that mimics the Hugging Face pipeline interface.
    """

    def __init__(self, score):
        self.score = score

    def __call__(self, text):
        return [{"label": "FAKE", "score": self.score}]


class ToxicModelApiTests(unittest.TestCase):
    def setUp(self):
        """
        Runs before each test.
        Creates a Flask test client and stores the original classifier.
        """
        app_module.app.testing = True
        self.client = app_module.app.test_client()
        self.original_clf = app_module.clf

    def tearDown(self):
        """
        Runs after each test.
        Restores the original classifier so tests stay isolated.
        """
        app_module.clf = self.original_clf

    def test_health_returns_ok(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["server response"], "ok")
        self.assertIn("model_name", data)
        self.assertIn("threshold", data)

    def test_predict_rejects_missing_text(self):
        response = self.client.post("/predict", json={})

        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertIn("error", data)

    def test_predict_returns_toxic_when_score_above_threshold(self):
        app_module.clf = FakeClassifier(score=0.95)

        response = self.client.post("/predict", json={
            "text": "You are disgusting and useless."
        })

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["label"], "toxic")
        self.assertAlmostEqual(data["toxic_score"], 0.95)
        self.assertEqual(data["threshold"], app_module.THRESHOLD)
        self.assertIn("model_name", data)

    def test_predict_returns_non_toxic_when_score_below_threshold(self):
        app_module.clf = FakeClassifier(score=0.01)

        response = self.client.post("/predict", json={
            "text": "Hello, how are you?"
        })

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["label"], "non-toxic")
        self.assertAlmostEqual(data["toxic_score"], 0.01)
        self.assertEqual(data["threshold"], app_module.THRESHOLD)
        self.assertIn("model_name", data)


if __name__ == "__main__":
    unittest.main()