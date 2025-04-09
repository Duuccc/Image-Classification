"""
Contains functionality for making predictions with a trained PyTorch model.
"""
import torch
from PIL import Image
import model_builder
import data_setup
from torchvision import transforms, datasets
from typing import Tuple, List, Dict
import os
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import argparse

def predict_image(
    image_path: str,
    model_path: str = "models/05_going_modular_script_mode_tinyvgg_model.pth",
    class_names: List[str] = None,
    image_size: Tuple[int, int] = (64, 64),
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[str, float, torch.Tensor]:
    """
    Makes a prediction on a target image using a trained PyTorch model.
    
    Args:
        image_path: Path to target image
        model_path: Path to saved model state_dict
        class_names: List of class names for the model
        image_size: Size to resize image to
        device: Target device to compute on
        
    Returns:
        Tuple containing:
        - Predicted class name
        - Confidence score
        - Transformed image tensor
    """
    # Load image and transform it
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(dim=0)
    
    # Load model
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=len(class_names) if class_names else 3
    ).to(device)
    
    # Load model state
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Make prediction
    with torch.inference_mode():
        # Add batch dimension and move to device
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        target_image_pred = model(image_tensor)
        
        # Get prediction probabilities
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        
        # Get prediction label
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        
        # Get prediction probability
        target_image_pred_prob = target_image_pred_probs[0][target_image_pred_label]
        
        # Get class name
        predicted_class = class_names[target_image_pred_label] if class_names else f"Class {target_image_pred_label.item()}"
        
    return predicted_class, target_image_pred_prob.item(), image_tensor




def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Image Classification Prediction")
    
    # Add arguments
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to the image to predict",
        default="data/pizza_steak_sushi/train/pizza/320570.jpg"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Path to the trained model",
        default="models/05_going_modular_script_mode_tinyvgg_model.pth"
    )
    
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    class_names = ['pizza', "steak", "sushi"]
    
    # Make prediction on single image
    print(f"Making prediction on image: {args.image}")
    predicted_class, confidence, image_tensor = predict_image(
        image_path=args.image,
        model_path=args.model,
        class_names=class_names,
        device=device
    )

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
    
if __name__ == "__main__":
    main()