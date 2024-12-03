import os
import glob
from tqdm import tqdm
from models.feature_extractor import FeatureExtractor

def extract_features_for_dataset(dataset_path: str, model_name: str = "clip", device: str = "cuda"):
    extractor = FeatureExtractor(model_name=model_name, device=device)

    sequences = os.listdir(dataset_path)
    for seq in sequences:
        seq_path = os.path.join(dataset_path, seq)
        rgb_path = os.path.join(seq_path, "rgb")
        output_path = os.path.join(seq_path, f"{model_name}_features")
        os.makedirs(output_path, exist_ok=True)

        image_files = sorted(glob.glob(os.path.join(rgb_path, "*.png")))
        output_files = [os.path.join(output_path, os.path.basename(f).replace(".png", ".pt")) for f in image_files]

        for i in tqdm(range(0, len(image_files), 16), desc=f"Processing {seq}"):
            batch_images = image_files[i:i+16]
            batch_outputs = output_files[i:i+16]
            extractor.extract_and_save_features(batch_images, batch_outputs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract features using a 2D foundational model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_name", type=str, default="clip", choices=["clip", "sam", "dino"], help="Model to use for feature extraction.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    args = parser.parse_args()

    extract_features_for_dataset(args.dataset_path, args.model_name, args.device)
