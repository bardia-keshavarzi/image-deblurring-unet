# src/data/blur_generator.py
import cv2
import numpy as np
from pathlib import Path
import random


def motion_blur_kernel(size, angle):
    """
    Create motion blur kernel
    
    Args:
        size: Kernel size (e.g., 15, 25, 35)
        angle: Motion direction in degrees (e.g., 0, 45, 90, 135)
    
    Returns:
        Motion blur kernel matrix
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    # Create a line at the given angle
    angle_rad = np.deg2rad(angle)
    for i in range(size):
        offset = int((i - center) * np.cos(angle_rad))
        x = center + offset
        y = center + int((i - center) * np.sin(angle_rad))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    
    # Normalize so sum equals 1
    return kernel / kernel.sum()


def gaussian_blur_kernel(size, sigma):
    """
    Create Gaussian blur kernel
    
    Args:
        size: Kernel size (e.g., 5, 9, 15)
        sigma: Standard deviation (e.g., 1.0 to 5.0)
    
    Returns:
        Gaussian blur kernel matrix
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / kernel.sum()


def defocus_blur_kernel(size):
    """
    Create defocus blur kernel (circular/disk shape)
    
    Args:
        size: Kernel size (e.g., 11, 21, 31)
    
    Returns:
        Defocus blur kernel matrix
    """
    kernel = np.zeros((size, size))
    center = size // 2
    radius = size // 2
    
    # Create circular mask
    y, x = np.ogrid[-center:size-center, -center:size-center]
    mask = x*x + y*y <= radius*radius
    kernel[mask] = 1
    
    return kernel / kernel.sum()
# Add after your defocus_blur_kernel function

def add_gaussian_noise(image, noise_level=5.0):
    """
    Add Gaussian noise to image for realism
    
    Real cameras have sensor noise, especially in low light.
    This makes our synthetic blur more realistic.
    
    Args:
        image: Input image
        noise_level: Noise standard deviation (3-10 typical)
    
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def apply_blur(image, kernel):
    """
    Apply blur kernel to image
    
    Args:
        image: Input image (numpy array)
        kernel: Blur kernel
    
    Returns:
        Blurred image
    """
    return cv2.filter2D(image, -1, kernel)


def generate_motion_blur(image, kernel_size, angle):
    """
    Generate motion-blurred version of image
    
    Args:
        image: Input sharp image
        kernel_size: Size of motion blur kernel
        angle: Direction of motion blur
    
    Returns:
        Motion-blurred image
    """
    kernel = motion_blur_kernel(kernel_size, angle)
    return apply_blur(image, kernel)


def generate_gaussian_blur(image, kernel_size, sigma):
    """
    Generate Gaussian-blurred version of image
    
    Args:
        image: Input sharp image
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation
    
    Returns:
        Gaussian-blurred image
    """
    kernel = gaussian_blur_kernel(kernel_size, sigma)
    return apply_blur(image, kernel)


def generate_defocus_blur(image, kernel_size):
    """
    Generate defocus-blurred version of image
    
    Args:
        image: Input sharp image
        kernel_size: Size of defocus kernel
    
    Returns:
        Defocus-blurred image
    """
    kernel = defocus_blur_kernel(kernel_size)
    return apply_blur(image, kernel)


def generate_blur_dataset(sharp_images_dir, output_dir, config):
    """
    Generate blurred training dataset from sharp images
    
    Args:
        sharp_images_dir: Directory containing sharp images
        output_dir: Directory to save blurred/sharp pairs
        config: Configuration object with blur parameters
    """
    from ..utils import load_image, save_image
    
    sharp_dir = Path(sharp_images_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    blurry_dir = output_path / "blurry"
    sharp_output_dir = output_path / "sharp"
    blurry_dir.mkdir(parents=True, exist_ok=True)
    sharp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of sharp images
    image_files = list(sharp_dir.glob("*.jpg")) + list(sharp_dir.glob("*.png"))
    
    print(f"Found {len(image_files)} sharp images")
    print(f"Generating blurred versions...")
    
    # Get blur types from config
    blur_types = config.get('blur.types', ['motion', 'gaussian', 'defocus'])
    
    # Initialize generator
    generator = BlurGenerator(blur_types=blur_types)
    
    # Process each image
    for idx, img_path in enumerate(image_files):
        # Load sharp image
        sharp_img = load_image(str(img_path), color_mode='RGB')
        
        # Generate multiple blurred versions (one per blur type)
        for blur_type in blur_types:
            # Use generator class
            blurry_img, params = generator.generate_blur(sharp_img, blur_type=blur_type)
            
            # Create descriptive filename
            if params['type'] == 'motion':
                blur_name = f"motion_k{params['kernel_size']}_a{params['angle']}"
            elif params['type'] == 'gaussian':
                blur_name = f"gaussian_k{params['kernel_size']}_s{params['sigma']:.1f}"
            elif params['type'] == 'defocus':
                blur_name = f"defocus_k{params['kernel_size']}"
            
            # Save blurred and sharp pair
            base_name = img_path.stem
            blurry_filename = f"{base_name}_{blur_name}.png"
            sharp_filename = f"{base_name}_{blur_name}.png"
            
            save_image(blurry_img, blurry_dir / blurry_filename, color_mode='RGB')
            save_image(sharp_img, sharp_output_dir / sharp_filename, color_mode='RGB')
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images")
    
    print(f"✓ Dataset generation complete!")
    print(f"  Blurred images: {blurry_dir}")
    print(f"  Sharp images: {sharp_output_dir}")
# Add at the end of blur_generator.py

class BlurGenerator:
    """
    Unified interface for blur generation
    
    Wraps individual blur functions for easy use in PyTorch datasets
    
    USAGE:
        generator = BlurGenerator(blur_types=['motion', 'gaussian', 'defocus'])
        blurred, params = generator.generate_blur(sharp_image)
    """
    
    def __init__(self, blur_types=['motion', 'gaussian', 'defocus'], add_noise=True):
        """
        Initialize blur generator
        
        Args:
            blur_types: List of blur types to use
            add_noise: Whether to add noise (30% chance)
        """
        self.blur_types = blur_types
        self.add_noise = add_noise
        
        print(f"✓ BlurGenerator initialized: {blur_types}")
    
    def generate_blur(self, image, blur_type=None):
        """
        Generate random blur on image
        
        THIS IS THE MAIN METHOD - use this in datasets
        
        Args:
            image: Sharp input image (RGB or BGR)
            blur_type: Specific blur type, or None for random
        
        Returns:
            Tuple of (blurred_image, parameters_dict)
        
        Example:
            >>> gen = BlurGenerator()
            >>> blurred, params = gen.generate_blur(sharp_img)
            >>> print(params['type'])  # 'motion', 'gaussian', or 'defocus'
        """
        # Choose random blur type if not specified
        if blur_type is None:
            blur_type = random.choice(self.blur_types)
        
        # Generate blur based on type
        if blur_type == 'motion':
            kernel_size = random.choice([15, 25, 35, 45])
            angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            blurred = generate_motion_blur(image, kernel_size, angle)
            params = {
                'type': 'motion',
                'kernel_size': kernel_size,
                'angle': angle
            }
        
        elif blur_type == 'gaussian':
            kernel_size = random.choice([5, 9, 15, 21])
            sigma = random.uniform(1.0, 5.0)
            blurred = generate_gaussian_blur(image, kernel_size, sigma)
            params = {
                'type': 'gaussian',
                'kernel_size': kernel_size,
                'sigma': sigma
            }
        
        elif blur_type == 'defocus':
            kernel_size = random.choice([11, 21, 31, 41])
            blurred = generate_defocus_blur(image, kernel_size)
            params = {
                'type': 'defocus',
                'kernel_size': kernel_size
            }
        
        else:
            raise ValueError(f"Unknown blur type: {blur_type}")
        
        # Add noise 30% of the time
        if self.add_noise and random.random() < 0.3:
            noise_level = random.uniform(3, 10)
            blurred = add_gaussian_noise(blurred, noise_level)
            params['noise_level'] = noise_level
        
        return blurred, params
    
    def generate_multiple(self, image, num_variations=5):
        """
        Generate multiple blurred versions of same image
        
        Args:
            image: Input sharp image
            num_variations: How many blurred versions to create
        
        Returns:
            List of (blurred_image, params) tuples
        """
        variations = []
        for i in range(num_variations):
            blurred, params = self.generate_blur(image)
            variations.append((blurred, params))
        
        return variations
