"""
Model Inference Module

This module handles model loading and inference for real-time sneeze detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from utils.model_definition import LightweightSneezeCNN
from utils.config import MODEL_PATH, DEVICE, THRESHOLD, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH


class ModelInferenceModule:
    """
    Model inference module for real-time sneeze detection

    Handles model loading, preprocessing of MFCC features, and prediction.
    """

    def __init__(self, model_path=MODEL_PATH, device=DEVICE, threshold=THRESHOLD):
        """
        Initialize model inference module

        Args:
            model_path (str): Path to trained model weights (.pth file)
            device (str): Device for inference ('cpu' or 'cuda')
            threshold (float): Detection threshold (default: 0.8)
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.threshold = threshold
        self.model = None

        # Load model
        self.load_model()

    def load_model(self):
        """
        Load trained model from file

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please ensure the model file exists at the specified path."
            )

        try:
            # Initialize model architecture
            self.model = LightweightSneezeCNN(
                input_height=MODEL_INPUT_HEIGHT,
                input_width=MODEL_INPUT_WIDTH,
                num_classes=2
            )

            # Load trained weights
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

            # Set to evaluation mode
            self.model.eval()

            # Move to device
            self.model = self.model.to(self.device)

            # Optimize for inference
            if self.device.type == 'cpu':
                torch.set_num_threads(2)  # Limit CPU threads for embedded systems

            print(f"✓ Model loaded successfully")
            print(f"  Path: {self.model_path}")
            print(f"  Device: {self.device}")
            print(f"  Parameters: {self.model.get_num_params():,}")
            print(f"  Threshold: {self.threshold}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, mfcc_features):
        """
        Predict sneeze probability from MFCC features

        Args:
            mfcc_features (np.ndarray): MFCC features with shape (60, 63)

        Returns:
            tuple: (is_sneeze, probability)
                - is_sneeze (bool): True if sneeze detected (prob > threshold)
                - probability (float): Sneeze probability [0, 1]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Validate input shape
        if mfcc_features.shape != (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH):
            raise ValueError(
                f"Invalid MFCC shape: {mfcc_features.shape}. "
                f"Expected ({MODEL_INPUT_HEIGHT}, {MODEL_INPUT_WIDTH})"
            )

        # Convert to tensor and add batch + channel dimensions
        # Shape: (60, 63) -> (1, 1, 60, 63)
        mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).unsqueeze(0)
        mfcc_tensor = mfcc_tensor.to(self.device)

        # Inference
        with torch.inference_mode():  # Faster than torch.no_grad()
            output = self.model(mfcc_tensor)  # Shape: (1, 2)

            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)

            # Extract sneeze probability (class 1)
            sneeze_prob = probabilities[0][1].item()

        # Decision based on threshold
        is_sneeze = sneeze_prob > self.threshold

        return is_sneeze, sneeze_prob

    def predict_batch(self, mfcc_batch):
        """
        Predict for a batch of MFCC features

        Args:
            mfcc_batch (list): List of MFCC feature arrays

        Returns:
            list: List of (is_sneeze, probability) tuples
        """
        results = []
        for mfcc in mfcc_batch:
            is_sneeze, prob = self.predict(mfcc)
            results.append((is_sneeze, prob))
        return results

    def get_model_info(self):
        """
        Get model information

        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "path": str(self.model_path),
            "device": str(self.device),
            "num_params": self.model.get_num_params(),
            "threshold": self.threshold,
            "input_shape": (1, 1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
            "output_shape": (1, 2)
        }


if __name__ == "__main__":
    # Test model inference module
    print("Testing ModelInferenceModule...")
    print("=" * 70)

    try:
        # Initialize model
        model_inference = ModelInferenceModule()

        # Print model info
        info = model_inference.get_model_info()
        print("\nModel Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test with dummy MFCC features
        print("\nTesting inference with dummy data...")
        dummy_mfcc = np.random.randn(60, 63).astype(np.float32)

        is_sneeze, probability = model_inference.predict(dummy_mfcc)

        print(f"\nPrediction:")
        print(f"  Is sneeze: {is_sneeze}")
        print(f"  Probability: {probability:.4f}")
        print(f"  Threshold: {model_inference.threshold}")

        print("\n✓ Model inference test passed!")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nNote: This test requires a trained model at:")
        print(f"  {MODEL_PATH}")
        print("\nPlease train the model first using sneeze_detection_lightweight.ipynb")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
