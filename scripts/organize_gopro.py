
"""
GOPRO_Large/
  train/
    GOPRxxxx_xx_xx/          # a scene/sequence folder
      blur/
        000001.png
        000002.png
        ...
      sharp/
        000001.png
        000002.png
        ...
    GOPRyyyy_yy_yy/
      blur/...
      sharp/...
  test/
    GOPRzzzz_zz_zz/
      blur/...
      sharp/...

"""
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm


def organize_gopro(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("Organizing GoPro Dataset")
    print("=" * 60)
    
    for split in ['train', 'test']:
        split_dir = input_dir / split
        
        if not split_dir.exists():
            print(f"{split_dir} not found, skipping")
            continue
        
        blur_out = output_dir / split / 'blurred'
        sharp_out = output_dir / split / 'sharp'
        blur_out.mkdir(parents=True, exist_ok=True)
        sharp_out.mkdir(parents=True, exist_ok=True)
        
        scenes = []
        for d in split_dir.iterdir():
            if d.is_dir():
                scenes.append(d)
        print(f"\n{split.upper()}: {len(scenes)} scenes")
        
        count = 0
        for scene in tqdm(scenes, desc=f"Processing {split}"):
            scene_blur = scene / 'blur'
            scene_sharp = scene / 'sharp'
            
            if not scene_blur.exists() or not scene_sharp.exists():
                continue
            
            blur_imgs = sorted(scene_blur.glob('*.png'))
            sharp_imgs = sorted(scene_sharp.glob('*.png'))
            
            for blur_img, sharp_img in zip(blur_imgs, sharp_imgs):
                new_name = f"{scene.name}_{blur_img.name}"
                shutil.copy2(blur_img, blur_out / new_name)
                shutil.copy2(sharp_img, sharp_out / new_name)
                count += 1
        
        print(f"Copied {count} pairs")
    
    print("\nDataset ready at:", output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to GOPRO_Large')
    parser.add_argument('--output', default='data/gopro', help='Output dir')
    args = parser.parse_args()
    organize_gopro(args.input, args.output)
