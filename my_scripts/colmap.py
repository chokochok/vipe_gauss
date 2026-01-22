import argparse
import pycolmap
from pathlib import Path
import os
from PIL import Image
from tqdm import tqdm
import shutil

def resize_images_in_folder(source_path: Path):
    images_dir = source_path / "images"
    if not images_dir.exists():
        print(f"❌ Error: Folder {images_dir} does not exist!")
        return

    scales = [2, 4, 8]
    for scale in scales:
        (source_path / f"images_{scale}").mkdir(exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in images_dir.iterdir() if f.suffix.lower() in valid_exts]

    print(f"🔄 Resizing {len(files)} images by factors {scales}...")

    for file_path in tqdm(files, desc="Resizing"):
        try:
            with Image.open(file_path) as img:
                W, H = img.size
                for scale in scales:
                    new_W, new_H = W // scale, H // scale
                    resized_img = img.resize((new_W, new_H), Image.Resampling.LANCZOS)
                    save_path = source_path / f"images_{scale}" / file_path.name
                    resized_img.save(save_path)
        except Exception as e:
            print(f"⚠️ Failed to process {file_path.name}: {e}")

def run_colmap(source_path: Path, matcher_type: str):
    images_dir = source_path / "images"
    database_path = source_path / "database.db"
    sparse_path = source_path / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    # 1. Feature Extraction
    print(f"\n🚀 [1/3] Feature Extraction...")
    
    # Визначаємо правильний клас опцій
    if hasattr(pycolmap, 'FeatureExtractionOptions'):
        sift_opt = pycolmap.FeatureExtractionOptions()
    else:
        sift_opt = pycolmap.SiftExtractionOptions()
        
    reader_opt = pycolmap.ImageReaderOptions()
    
    try:
        pycolmap.extract_features(
            str(database_path), 
            str(images_dir), 
            reader_options=reader_opt,
            extraction_options=sift_opt
        )
    except TypeError:
        pycolmap.extract_features(
            str(database_path), 
            str(images_dir), 
            sift_options=sift_opt
        )

    # 2. Matching
    print(f"\n🤝 [2/3] Matching (Mode: {matcher_type})...")
    
    if matcher_type == "sequential":
        # === ВИПРАВЛЕННЯ ТУТ ===
        # Створюємо об'єкт опцій для sequential matching
        pairing_opt = pycolmap.SequentialPairingOptions()
        pairing_opt.overlap = 20  # Встановлюємо overlap всередині об'єкта
        
        pycolmap.match_sequential(
            str(database_path), 
            pairing_options=pairing_opt  # Передаємо об'єкт, а не аргумент overlap
        )
    else:
        pycolmap.match_exhaustive(str(database_path))

    # 3. Mapping
    print(f"\n🏗️  [3/3] Reconstruction (Mapping)...")
    
    maps = pycolmap.incremental_mapping(str(database_path), str(images_dir), str(sparse_path))

    if not maps:
        print("❌ Error: Failed to create a 3D model.")
        return

    best_reconstruction = maps[0]
    
    print("-" * 50)
    print(f"✅ Done! Results saved to: {source_path}")
    print(f"📊 Cameras: {best_reconstruction.num_reg_images()} | Points: {best_reconstruction.num_points3D()}")

def main():
    parser = argparse.ArgumentParser(description="Prepare scene for Gaussian Splatting")
    parser.add_argument("source_path", type=Path, help="Project folder with 'images' inside")
    parser.add_argument("--resize", action="store_true", help="Resize images")
    parser.add_argument("--matcher", type=str, choices=["sequential", "exhaustive"], default="sequential")
    args = parser.parse_args()

    if not args.source_path.exists() or not (args.source_path / "images").exists():
        print(f"❌ Error: Check your path: {args.source_path}")
        return

    if args.resize:
        resize_images_in_folder(args.source_path)

    run_colmap(args.source_path, args.matcher)

if __name__ == "__main__":
    main()