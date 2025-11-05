import onnx
import time
import numpy as np
import torch as th
from typing import Tuple
import onnxruntime as ort
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

# Define ONNX Policy
class OnnxableSB3Policy(th.nn.Module):
    """
    Thin wrapper to make a Stable Baselines3 policy callable like a nn.Module.
    This allows exporting the SB3 policy to ONNX via torch.onnx.export.
    The wrapper forwards the observation to the underlying SB3 policy and
    requests deterministic output (useful for inference/export).
    """
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # The SB3 policy returns (actions, values, log_probs) when called.
        # Keep the same signature so ONNX export captures the raw policy outputs.
        return self.policy(observation, deterministic=True)

# Load SB3 model and export to ONNX
def export_onnx(test=False):
    """
    Load a trained PPO model (path hardcoded here) and export its policy to ONNX.
    If test=True, run a simple consistency loop comparing ONNX outputs vs SB3 outputs.
    """
    # Load the trained model onto CPU
    model = PPO.load("Models/Angular/angular_model_5.zip", device="cpu")

    # Wrap the SB3 policy so it can be exported with torch.onnx.export
    onnx_policy = OnnxableSB3Policy(model.policy)

    # Get observation shape from the model's observation space (e.g., (3,240,320))
    observation_size = model.observation_space.shape

    # Create a dummy input tensor with a batch dimension for ONNX export.
    # Adjust size to match model expectations: here hardcoded to (1,3,240,320).
    # If your model expects a different shape, change this accordingly.
    dummy_input = th.randn(size=(1,3,240,320))

    # Export the wrapped policy to ONNX. Set an appropriate opset_version.
    # The input name "obs" will be used by ONNX Runtime to feed inputs.
    th.onnx.export(
        onnx_policy,
        dummy_input,
        "angular_model_rgb.onnx",
        opset_version=17,
        input_names=["obs"],
    )

    # Optional testing block: validate exported model and compare runtime outputs
    if test:
        onnx_path = "angular_model_rgb.onnx"

        # Basic sanity check on the ONNX file structure
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Create an ONNX Runtime session for inference
        ort_sess = ort.InferenceSession(onnx_path)

        # Continuously compare outputs from ONNX runtime to SB3 policy for random inputs
        while True:
            # Generate a random observation matching the original observation_size.
            # Ensure dtype is float32 (ONNX Runtime expects numpy arrays).
            obs = np.random.randn(1, *observation_size).astype(np.float32)

            # Run ONNX model. The model was exported to accept a single input named "obs".
            onnx_actions, _, _ = ort_sess.run(None, {"obs": obs})

            # Run SB3 policy directly for comparison (raw policy outputs, not postprocessed).
            sb3_raw_actions, _, _ = model.policy(th.as_tensor(obs), deterministic=True)

            # Also compare against the SB3 higher-level predict() which may apply additional processing.
            sb3_predict_actions, _ = model.predict(obs, deterministic=True)

            # Print results for manual inspection. In production you'd replace this with automated checks.
            print("ONNX raw:", onnx_actions)
            print("SB3 policy raw:", sb3_raw_actions)
            print("SB3 predict:", sb3_predict_actions)
            print("\n")
            time.sleep(0.1)

if __name__ == "__main__":
    # Run export when executed as a script. Set test=True to enable the validation loop.
    export_onnx()