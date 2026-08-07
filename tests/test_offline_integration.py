import os
import unittest

from scripts.smokeOfflinePipeline import runOfflineSmoke


@unittest.skipUnless(
    os.environ.get("TALONCV_RUN_OFFLINE_SMOKE") == "1",
    "Set TALONCV_RUN_OFFLINE_SMOKE=1 after explicit local model setup.",
)
class OfflineIntegrationTests(unittest.TestCase):
    def test_all_real_models_and_full_pipeline_with_sockets_blocked(self):
        result = runOfflineSmoke(maxCoachTokens=32)
        self.assertTrue(result["ready"])
        self.assertEqual([], result["networkAttempts"])
        self.assertGreater(result["localCoachCharacters"], 0)
        self.assertGreater(result["reportCharacters"], 1000)


if __name__ == "__main__":
    unittest.main()
