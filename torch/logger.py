"""
Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO         


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer =  tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(name=tag, data=value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert the image to a PNG-encoded string
                buffer = BytesIO()
                if len(img.shape) == 2:  # Grayscale
                    mode = 'L'
                elif len(img.shape) == 3 and img.shape[2] == 3:  # RGB
                    mode = 'RGB'
                else:
                    raise ValueError(f"Unsupported image shape: {img.shape}")

                Image.fromarray(np.uint8(img), mode).save(buffer, format="PNG")
                png_encoded = buffer.getvalue()

                # Log the image
                tf.summary.image(f"{tag}/{i}", np.expand_dims(img, axis=(0, -1)), step=step)
            self.writer.flush()
        
    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            # Create histogram using NumPy
            counts, bin_edges = np.histogram(values, bins=bins)

            # Compute midpoints of bins for visualization
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

            # Log histogram using TensorBoard's scalar logging
            for i, count in enumerate(counts):
                tf.summary.scalar(f"{tag}/bin_{i}", count, step=step)

            # Optionally log histogram data
            tf.summary.histogram(name=tag, data=values, step=step)
            self.writer.flush()