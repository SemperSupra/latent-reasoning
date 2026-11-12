"""Credentialless surface conformance for the exact Kaggle SDK runtime pinned by CI.

No API methods are invoked. This is not a quota observation, notebook
submission, device qualification, or authenticated access test.
"""
import inspect
import unittest


class KaggleSDKSurfaceTests(unittest.TestCase):
    def test_sdk_version_and_typed_method_presence(self):
        from kaggle import api
        for name in ("authenticate", "get_config_value", "quota_view",
                     "kernels_list_with_response", "kernels_status",
                     "kernels_push", "kernels_output", "build_kaggle_client"):
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(api, name, None)), name)
        self.assertTrue(hasattr(api, "CONFIG_NAME_USER"))

    def test_kernels_push_timeout_acc_and_metadata(self):
        from kaggle import api
        sig = inspect.signature(api.kernels_push)
        self.assertIn("folder", sig.parameters)
        self.assertIn("timeout", sig.parameters)
        self.assertIn("acc", sig.parameters)

    def test_native_version_sdk_request_type(self):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        req = ApiGetKernelRequest()
        req.user_name = "synthetic"
        req.kernel_slug = "synthetic"
        self.assertEqual(req.user_name, "synthetic")
        self.assertEqual(req.kernel_slug, "synthetic")

    def test_list_and_status_names_present_offline(self):
        from kaggle import api
        self.assertTrue(callable(api.kernels_list_with_response))
        self.assertTrue(callable(api.kernels_status))
        self.assertTrue(callable(api.kernels_output))

if __name__ == "__main__":
    unittest.main()
