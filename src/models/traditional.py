# src/models/traditional.py
"""
Traditional Deblurring Methods
These are classical computer vision techniques (no deep learning)
"""

import cv2
import numpy as np
from scipy import signal
import logging

# Setup logging so we can see what's happening
logger = logging.getLogger(__name__)


class TraditionalDeblurrer:
    """
    Collection of traditional deblurring methods
    
    Think of this as your toolbox with 4 different tools:
    1. Unsharp Masking - Simple and fast
    2. Simple Sharpening - Very simple and very fast
    3. Wiener Filter - Good when you know the blur kernel
    4. Richardson-Lucy - Iterative, can produce good results
    
    Usage:
        deblurrer = TraditionalDeblurrer()
        sharp_image = deblurrer.unsharp_masking(blurred_image)
    """
    
    def __init__(self):
        """Initialize the deblurrer"""
        # Dictionary mapping method names to functions
        self.methods = {
            'unsharp_mask': self.unsharp_masking,
            'simple_sharpen': self.simple_sharpening,
            'wiener': self.wiener_filter,
            'richardson_lucy': self.richardson_lucy
        }
        logger.info("TraditionalDeblurrer initialized with 4 methods")

# Add to TraditionalDeblurrer class

    def unsharp_masking(self, image, sigma=1.0, strength=1.5):
        """
        Unsharp masking - sharpens by enhancing edges
        
        HOW IT WORKS:
        1. Create blurred version of image
        2. Calculate difference: detail = image - blurred
        3. Add amplified detail back: sharp = image + strength * detail
        
        WHEN TO USE:
        - General purpose sharpening
        - When you don't know what caused the blur
        - Quick enhancement needed
        
        Args:
            image: Blurred image (BGR or grayscale)
            sigma: Blur amount (0.5=sharp, 3.0=smooth)
            strength: How much to sharpen (0.5=subtle, 2.0=strong)
        
        Returns:
            Sharpened image
        
        Example:
            >>> sharp = deblurrer.unsharp_masking(blurred, sigma=1.0, strength=1.5)
        """
        try:
            logger.info(f"Applying unsharp mask: sigma={sigma}, strength={strength}")
            
            # Calculate kernel size from sigma
            # Rule: kernel = 2 * (3 * sigma) + 1 (must be odd)
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Make it odd
            
            # Step 1: Blur the image
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
            # Step 2 & 3: Add amplified detail
            # This formula does: image + strength * (image - blurred)
            sharpened = cv2.addWeighted(
                image, 1.0 + strength,  # Original with extra weight
                blurred, -strength,      # Subtract blur
                0                        # No offset
            )
            
            # Clip values to valid range [0, 255]
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            logger.info("✓ Unsharp masking completed")
            return sharpened
            
        except Exception as e:
            logger.error(f"Unsharp masking failed: {e}")
            return image  # Return original if failed

# Add to TraditionalDeblurrer class

    def simple_sharpening(self, image, strength=1.0):
        """
        Simple sharpening using convolution kernel
        
        HOW IT WORKS:
        - Uses a 3x3 kernel that emphasizes center pixel
        - Subtracts neighbors, amplifies center
        - Kernel: [[-1 -1 -1]
                   [-1  9 -1]
                   [-1 -1 -1]]
        
        WHEN TO USE:
        - Need very fast sharpening
        - Slight blur only
        - Real-time applications
        
        Args:
            image: Blurred image
            strength: Sharpening strength (0.5=gentle, 2.0=aggressive)
        
        Returns:
            Sharpened image
        
        Example:
            >>> sharp = deblurrer.simple_sharpening(blurred, strength=1.0)
        """
        try:
            logger.info(f"Applying simple sharpening: strength={strength}")
            
            # Create sharpening kernel
            # Center = 8 + strength (amplifies center pixel)
            # Neighbors = -1 (subtracts from neighbors)
            kernel = np.array([
                [-1, -1, -1],
                [-1, 8 + strength, -1],
                [-1, -1, -1]
            ]) / (strength + 1)  # Normalize
            
            # Apply kernel to image
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Clip to valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            logger.info("✓ Simple sharpening completed")
            return sharpened
            
        except Exception as e:
            logger.error(f"Simple sharpening failed: {e}")
            return image

# Add to TraditionalDeblurrer class in src/models/traditional.py

    def wiener_filter(self, blurred_image, kernel=None, noise_variance=0.01):
        """
        Wiener filter - frequency domain deconvolution
        
        HOW IT WORKS:
        1. Convert image and kernel to frequency domain (FFT)
        2. Apply Wiener formula: output = input * H* / (|H|² + noise)
        3. Convert back to spatial domain
        
        WHERE: H = blur kernel, H* = complex conjugate
        
        WHEN TO USE:
        - You KNOW what kernel caused the blur
        - Image has some noise
        - Want theoretically optimal restoration
        
        Args:
            blurred_image: Input blurred image
            kernel: Blur kernel (PSF). If None, creates default motion blur
            noise_variance: Noise level estimate (0.001-0.1)
                        Higher = assume more noise = less aggressive
        
        Returns:
            Deblurred image
        
        Example:
            >>> # Create motion blur kernel
            >>> kernel = motion_blur_kernel(15, 0)  # From blur_generator
            >>> sharp = deblurrer.wiener_filter(blurred, kernel, noise_variance=0.01)
        """
        try:
            logger.info(f"Applying Wiener filter: noise_variance={noise_variance}")
            
            # If no kernel provided, create default motion blur
            if kernel is None:
                # Import from blur_generator where all kernels live
                from ..data.blur_generator import motion_blur_kernel
                kernel = motion_blur_kernel(15, 0)  # Horizontal motion blur
                logger.info("Using default horizontal motion blur kernel (15px, 0°)")
            
            # Normalize kernel (sum = 1)
            kernel = kernel.astype(np.float64)
            kernel /= np.sum(kernel)
            
            # Process each color channel separately
            if len(blurred_image.shape) == 3:
                result = np.zeros_like(blurred_image)
                for c in range(3):  # For R, G, B
                    result[:, :, c] = self._wiener_single_channel(
                        blurred_image[:, :, c],
                        kernel,
                        noise_variance
                    )
                logger.info("✓ Wiener filter applied to RGB channels")
            else:
                # Grayscale
                result = self._wiener_single_channel(
                    blurred_image,
                    kernel,
                    noise_variance
                )
                logger.info("✓ Wiener filter applied to grayscale")
            
            # Clip to valid range
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"Wiener filter failed: {e}")
            return blurred_image

# Add to TraditionalDeblurrer class

    def _wiener_single_channel(self, image, kernel, noise_var):
        """
        Apply Wiener filter to single channel
        
        This does the actual Wiener filtering math
        """
        # Convert to float [0, 1]
        img_float = image.astype(np.float64) / 255.0
        
        # Pad image to reduce edge effects
        pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
        padded_img = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # Create padded kernel (same size as image)
        padded_kernel = np.zeros_like(padded_img)
        padded_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel
        
        # Shift kernel to center (for proper FFT)
        padded_kernel = np.roll(
            padded_kernel,
            (-kernel.shape[0]//2, -kernel.shape[1]//2),
            axis=(0, 1)
        )
        
        # === FREQUENCY DOMAIN PROCESSING ===
        # Transform to frequency domain
        img_fft = np.fft.fft2(padded_img)
        kernel_fft = np.fft.fft2(padded_kernel)
        
        # Wiener filter formula:
        # W = H* / (|H|² + noise)
        kernel_conj = np.conj(kernel_fft)  # Complex conjugate
        kernel_mag_sq = np.abs(kernel_fft) ** 2  # |H|²
        
        epsilon = 1e-10  # Prevent division by zero
        wiener_filter = kernel_conj / (kernel_mag_sq + noise_var + epsilon)
        
        # Apply filter
        result_fft = img_fft * wiener_filter
        
        # Transform back to spatial domain
        result = np.real(np.fft.ifft2(result_fft))
        
        # Remove padding
        result = result[pad_h:-pad_h, pad_w:-pad_w]
        
        # Convert back to uint8 [0, 255]
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result

# Add to TraditionalDeblurrer class in src/models/traditional.py

    def richardson_lucy(self, blurred_image, kernel=None, iterations=30):
        """
        Richardson-Lucy deconvolution - iterative restoration
        
        HOW IT WORKS:
        Starts with guess (blurred image) and improves it iteratively:
        1. estimate = blurred_image
        2. For N iterations:
            - Simulate blur: convolved = estimate ⊗ kernel
            - Calculate error: ratio = blurred / convolved
            - Correct estimate: estimate *= (ratio ⊗ kernel_flipped)
        
        WHEN TO USE:
        - Have time for iterations (slower than other methods)
        - Know the blur kernel
        - Want best possible quality
        
        Args:
            blurred_image: Input blurred image
            kernel: Blur kernel. If None, uses default
            iterations: Number of iterations (10-50)
                    More = better quality but slower
        
        Returns:
            Deblurred image
        
        Example:
            >>> kernel = defocus_blur_kernel(21)  # From blur_generator
            >>> sharp = deblurrer.richardson_lucy(blurred, kernel, iterations=30)
        """
        try:
            logger.info(f"Starting Richardson-Lucy: {iterations} iterations")
            
            # Default kernel if not provided
            if kernel is None:
                # CORRECTED: Import from blur_generator where kernels actually live
                from ..data.blur_generator import motion_blur_kernel
                kernel = motion_blur_kernel(15, 0)
                logger.info("Using default horizontal motion blur kernel (15px, 0°)")
            
            # Normalize inputs to [0, 1]
            blurred = blurred_image.astype(np.float64) / 255.0
            kernel = kernel.astype(np.float64)
            kernel /= np.sum(kernel)  # Ensure kernel sum = 1
            
            # Process each channel separately
            if len(blurred.shape) == 3:
                # RGB image
                result = np.zeros_like(blurred)
                for c in range(3):
                    logger.info(f"Processing channel {c+1}/3...")
                    result[:, :, c] = self._richardson_lucy_single_channel(
                        blurred[:, :, c],
                        kernel,
                        iterations
                    )
                logger.info("✓ All RGB channels processed")
            else:
                # Grayscale image
                result = self._richardson_lucy_single_channel(
                    blurred,
                    kernel,
                    iterations
                )
                logger.info("✓ Grayscale channel processed")
            
            # Convert back to uint8 [0, 255]
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            
            logger.info(f"✓ Richardson-Lucy completed: {iterations} iterations")
            return result
            
        except Exception as e:
            logger.error(f"Richardson-Lucy failed: {e}")
            return blurred_image
# Add to TraditionalDeblurrer class

    def _richardson_lucy_single_channel(self, image, kernel, iterations):
        """
        Richardson-Lucy for single channel
        
        This is where the iterative magic happens
        """
        # Start with estimate = input
        estimate = image.copy()
        estimate = np.maximum(estimate, 1e-10)  # Avoid zeros (cause division problems)
        
        # Flip kernel for convolution
        kernel_flipped = np.flipud(np.fliplr(kernel))
        
        # Iteratively improve estimate
        for i in range(iterations):
            # Simulate what blur would look like with current estimate
            convolved = signal.convolve2d(estimate, kernel, mode='same', boundary='symm')
            convolved = np.maximum(convolved, 1e-10)  # Avoid division by zero
            
            # Calculate error ratio
            ratio = image / convolved
            
            # Propagate error back
            correction = signal.convolve2d(ratio, kernel_flipped, mode='same', boundary='symm')
            
            # Update estimate
            estimate *= correction
            
            # Ensure positivity (can't have negative light!)
            estimate = np.maximum(estimate, 0)
            
            # Log progress every 10 iterations
            if (i + 1) % 10 == 0:
                logger.debug(f"  Iteration {i+1}/{iterations}")
        
        return estimate
