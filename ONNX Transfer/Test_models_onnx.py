#!/usr/bin/env python

"""Simple tester for an exported ONNX RL policy using Eyesim camera.

This script loads an ONNX model and repeatedly feeds live camera frames to
the model, then applies the predicted action to the robot (via the provided
`eye` helper functions). The file intentionally keeps behavior identical to
the original program; comments explain assumptions and data shapes expected
by the ONNX model.
"""

import onnx
from eye import *  # project-specific helpers: camera, LCD, keys, PSD, VWSetSpeed
import numpy as np
import onnxruntime as ort

# Constants for camera settings
# These map to values provided by `eye` (QQVGA/QVGA constants, widths/heights).
# Keep channel ordering in mind: the ONNX model expects channel-first input
# (C, H, W) after we convert the raw image.
CAM_SETTING = QVGA
CAMWIDTH = QVGA_X
CAMHEIGHT = QVGA_Y
CHANNELS = 3

# Load ONNX model and create an inference session. The checker validates the
# model structure (useful to catch export errors early).
onnx_path = "angular_model.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
# ONNX Runtime session used to run inference on numpy inputs.
ort_sess = ort.InferenceSession(onnx_path)

safe = 300  # distance threshold used with PSD sensors to avoid collisions

# Tests onnx models
def test_models():
    """Loop reading camera frames, running ONNX inference, and applying actions.

    The loop continues until the user presses the stop key. Each iteration:
    - updates a simple LCD menu,
    - grabs a camera observation via `eyesim_get_observation()`,
    - runs the ONNX model to get an action, clamps the action to [-1,1],
    - scales and applies the action to the robot if PSD sensors report a safe
      distance (the `safe` threshold), and
    - exits cleanly when the stop key is pressed.
    """

    # Continue testing the loaded model until the user decides to stop
    while True:
        LCDMenu("-", "-", "-", "Stop")
        LCDSetPrintf(0, 60, "Testing Models")
        key = KEYRead()

        # Retrieve a single preprocessed observation compatible with the
        # exported ONNX model (shape: (1, C, H, W), dtype float32).
        onnx_image = eyesim_get_observation()

        # Run inference on the ONNX model. The exported model expects an input
        # named "obs" (this name comes from the export step) and returns the
        # raw policy outputs. `ort_sess.run` returns a list of outputs; for
        # this model the first output contains the action values.
        onnx_actions, _, _ = ort_sess.run(None, {"obs": onnx_image})

        # Extract scalar action from returned array and clamp to [-1, 1]. The
        # indexing assumes the ONNX output shape is (1, 1, N) or similar;
        # the original code used [0][0], so we keep the same indexing to avoid
        # behavioral changes.
        onnx_actions = onnx_actions[0][0]
        onnx_actions = np.clip(onnx_actions, -1.0, 1.0)
        print(f" Onnx Action: {onnx_actions}")

        # Apply angular speed scaled by the action. The PSD sensor checks
        # provide a simple safety (avoid applying speed if obstacles are close).
        if PSDGet(PSD_RIGHT) < safe or PSDGet(PSD_LEFT) < safe or PSDGet(PSD_RIGHT) < safe:
            VWSetSpeed(200, round(200 * onnx_actions))

        # End testing if user presses the stop key
        if key == KEY4:
            VWSetSpeed(0, 0)
            LCDClear()
            LCDSetPrintf(0, 60, "Testing Stopped")
            break

# Get raw camera image and convert to correct format for onnx model
def eyesim_get_observation():
        """Capture a frame from Eyesim and return a preprocessed numpy tensor.

        Processing steps:
        - capture the raw image buffer from `CAMGet()` (project-supplied C buffer),
        - display it on the LCD for debugging, then
        - convert the raw bytes into a numpy array, reshape to (H, W, C),
        - cast to float32, transpose to channel-first (C, H, W), and add a
            leading batch dimension to produce shape (1, C, H, W).

        The returned array is ready to feed to ONNX Runtime (dtype float32).
        """

        # Get image from camera and show it on the LCD for debugging/visibility.
        image = CAMGet()
        LCDImage(image)

        # Convert the raw C buffer to a NumPy array of bytes and reshape to
        # (height, width, channels). The code assumes `image` is a contiguous
        # buffer with exactly CAMHEIGHT * CAMWIDTH * CHANNELS bytes.
        image_np = np.frombuffer(image, dtype=np.uint8)
        image_np = image_np.reshape((CAMHEIGHT, CAMWIDTH, CHANNELS))

        # Convert to float32 for model input and move channels to the first
        # dimension as the ONNX model expects channel-first tensors.
        image_np = image_np.astype(np.float32)
        image_np = np.transpose(image_np, (2, 0, 1))  # channel-first

        # Add batch dimension: ONNX Runtime expects shape (1, C, H, W).
        onnx_image = np.expand_dims(image_np, axis=0)  # shape: (1, 3, H, W)

        return onnx_image

def main():
    # Initialize the camera with QQVGA resolution (160x120)
    CAMInit(CAM_SETTING) 
    LCDImageStart(0,0,CAMWIDTH,CAMHEIGHT)

    LCDSetPrintf(0,60,"Program Start")
    while True:
        LCDMenu("Test", "-", "-", "Quit")
        key = KEYRead()

        # Test the models
        if key == KEY1: 
            test_models()
                
        # Quit program
        elif key == KEY4:
            LCDClear()
            break

if __name__=="__main__":
    main()