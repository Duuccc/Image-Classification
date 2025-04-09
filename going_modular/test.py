import utils, model_builder, data_setup
import torch

from torchvision import transforms, datasets


def main():
    BATCH_SIZE = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=3
    ).to(device)

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    model.load_state_dict(torch.load(f="models/05_going_modular_script_mode_tinyvgg_model.pth"))
    
    predicts = utils.predictor(model=model, image_path="data/pizza_steak_sushi/train/pizza/320570.jpg", 
                               device=device, transform=data_transform, class_names=class_names)

    print(predicts)

if __name__ == '__main__':
    main()