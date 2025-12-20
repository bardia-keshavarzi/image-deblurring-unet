import torch
import numpy as np
from pathlib import Path
import cv2


class DeblurPredictor:

    def __init__(self, model_path: str, device: str = 'cuda'):

        from src.models.unet import UNet
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        

        self.model = UNet(in_channels=3, out_channels=3, use_attention=True)
        

        checkpoint = torch.load(model_path, map_location=self.device)
        

        if isinstance(checkpoint, dict):

            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"  Best PSNR: {checkpoint.get('best_psnr', 'unknown'):.2f} dB")
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"  Best PSNR: {checkpoint.get('best_psnr', 'unknown'):.2f} dB")
            else:

                self.model.load_state_dict(checkpoint)
        else:

            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def predict(self, blurred_image: np.ndarray) -> np.ndarray:

        input_tensor = self._preprocess(blurred_image)
        

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        

        deblurred = self._postprocess(output_tensor)
        
        return deblurred
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:

        image_float = image.astype(np.float32) / 255.0
        

        tensor = torch.from_numpy(image_float).permute(2, 0, 1)
        

        tensor = tensor.unsqueeze(0)
        

        tensor = tensor.to(self.device)
        
        return tensor
    
    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:

        tensor = tensor.squeeze(0)
        

        tensor = tensor.cpu()

        image = tensor.permute(1, 2, 0).numpy()

        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        
        return image
    
    def predict_file(self, input_path: str, output_path: str):

        blurred = cv2.imread(input_path)
        if blurred is None:
            raise ValueError(f"Cannot load image: {input_path}")
        

        blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        

        deblurred_rgb = self.predict(blurred_rgb)
        

        deblurred_bgr = cv2.cvtColor(deblurred_rgb, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(output_path, deblurred_bgr)
        print(f"Saved: {output_path}")

    def predict_tta(self, blurred_image: np.ndarray, num_augs: int = 4) -> np.ndarray:

        predictions = []
        
        # Original
        predictions.append(self.predict(blurred_image))
        
        if num_augs >= 2:

            flipped_h = np.fliplr(blurred_image)
            pred_h = self.predict(flipped_h)
            predictions.append(np.fliplr(pred_h))
        
        if num_augs >= 4:

            flipped_v = np.flipud(blurred_image)
            pred_v = self.predict(flipped_v)
            predictions.append(np.flipud(pred_v))
            

            flipped_hv = np.flipud(np.fliplr(blurred_image))
            pred_hv = self.predict(flipped_hv)
            predictions.append(np.flipud(np.fliplr(pred_hv)))
        
        if num_augs >= 8:

            for k in [1, 2, 3]:
                rotated = np.rot90(blurred_image, k)
                pred_rot = self.predict(rotated)
                predictions.append(np.rot90(pred_rot, -k))
        

        result = np.mean(predictions, axis=0).astype(np.uint8)
        return result    


# Test
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    print("Testing DeblurPredictor...")
 
    model_path = 'checkpoints/best_model.pth'
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Train a model first!")
        sys.exit(1)
    

    predictor = DeblurPredictor(model_path)

    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = predictor.predict(test_image)
    
    print(f"Input shape: {test_image.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output range: [{result.min()}, {result.max()}]")
    
    print("\nPredictor works!")
    print("\nUsage:")
    print("  predictor = DeblurPredictor('checkpoints/best_model.pth')")
    print("  deblurred = predictor.predict(blurred_image)")
