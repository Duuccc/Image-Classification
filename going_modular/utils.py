# %%writefile going_modular/utils.py
"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

from torchvision import datasets, transforms
from PIL import Image
from typing import List

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  
def predictor(model: torch.nn.Module,
              image_path: str,
              device: torch.device,
              transform: transforms.Compose,
              class_names: List[str]):
  """
  Uses a trained model to predict the class of an image.
  """
  model.eval()
  with torch.inference_mode():
    image = Image.open(image_path)
    image_transformed = transform(image)
    image_transformed = image_transformed.unsqueeze(dim=0)
    image_transformed = image_transformed.to(device)
    model.to(device)
    pred_probs = torch.softmax(model(image_transformed), dim=1)
    pred_label = torch.argmax(pred_probs, dim=1)
    pred_label = class_names[pred_label]
    return pred_label
  
