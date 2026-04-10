import io
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import app as app_module


class SegmentationApiTests(unittest.TestCase):
    def setUp(self):
        app_module.app.testing = True
        self.client = app_module.app.test_client()

    def make_test_image_bytes(self):
        image = Image.new("RGB", (128, 128), color=(255, 255, 255))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    def test_health_returns_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["task"], "house-segmentation")
        self.assertIn("device", data)
        self.assertIn("checkpoint_exists", data)

    def test_predict_rejects_missing_image_field(self):
        response = self.client.post("/predict", data={})
        self.assertEqual(response.status_code, 400)

        data = response.get_json()
        self.assertIn("error", data)

    @patch("app.get_model")
    def test_predict_returns_segmentation_summary(self, mock_get_model):
        fake_model = MagicMock()

        import torch
        fake_output = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        fake_output[:, :, 50:100, 60:120] = 10.0

        fake_model.return_value = {"out": fake_output}
        mock_get_model.return_value = fake_model

        image_bytes = self.make_test_image_bytes()

        response = self.client.post(
            "/predict",
            data={"image": (image_bytes, "sample.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["message"], "Segmentation completed")
        self.assertEqual(data["mask_width"], 256)
        self.assertEqual(data["mask_height"], 256)
        self.assertIn("house_pixels", data)
        self.assertIn("house_ratio", data)
        self.assertIn("saved_mask_path", data)
        self.assertIn("saved_overlay_path", data)


if __name__ == "__main__":
    unittest.main()