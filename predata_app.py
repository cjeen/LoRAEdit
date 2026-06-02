import torch
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# torch.multiprocessing.set_start_method("spawn")

# Global variables to store Florence model
florence_model = None
florence_processor = None

import datetime
import os
import subprocess
import shutil
import time
from pathlib import Path

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np
from PIL import Image

from loguru import logger as guru

from sam2.build_sam import build_sam2_video_predictor

# Florence model import - required dependency
from transformers import AutoProcessor, AutoModelForCausalLM


def init_florence_model():
    """Initialize Florence model, only needs to be called once"""
    global florence_model, florence_processor
        
    if florence_model is not None and florence_processor is not None:
        return True  # Model already loaded, no need to reload
    
    print("Loading Florence model, please wait...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load model and processor
    florence_model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    
    florence_processor = AutoProcessor.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True
    )
    print("Florence model loaded successfully")
    return True

def generate_caption(image_path, concept_prefix=""):
    """Use Florence model to generate caption for image"""
    global florence_model, florence_processor
    
    if florence_model is None or florence_processor is None:
        raise RuntimeError("Florence model not initialized, please call init_florence_model() first")
    
    device = next(florence_model.parameters()).device
    torch_dtype = next(florence_model.parameters()).dtype
    
    # Read image
    image = Image.open(image_path).convert("RGB")
    prompt = "<DETAILED_CAPTION>"

    # Construct input
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    # Generate caption
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Post-processing
    parsed_answer = florence_processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
    
    # Add concept prefix
    if concept_prefix:
        caption_text = f"{concept_prefix} {caption_text}"

    return caption_text


def normalize_training_precision(precision):
    precision = (precision or "").lower()
    if precision == "fp16":
        precision = "bf16"
    if precision not in ("bf16", "fp8"):
        raise ValueError(f"Unsupported training precision: {precision}. Choose 'bf16' or 'fp8'.")
    return precision


class PromptGUI(object):
    def __init__(self, checkpoint_dir, model_cfg):
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        self.sam_model = None
        self.tracker = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        # simplified to handle only one mask
        self.cur_mask = None
        self.cur_logit = None
        self.masks_all = []  # store binary masks for all frames
        self.bbox_masks_all = []  # store bbox masks for all frames

        self.img_dir = ""
        self.img_paths = []
        self.video_name = None  # Used to save recognized video name
        self.init_sam_model()

    def extract_video_name(self, path):
        """Intelligently recognize video name from path"""
        if not path:
            return "sequence"
            
        # If path contains video_frames_, it means extracted from video
        if "video_frames_" in path:
            # Use timestamp as video name
            import re
            match = re.search(r'video_frames_(\d+)', path)
            if match:
                timestamp = match.group(1)
                return f"video_{timestamp}"
        
        # Otherwise use directory name
        path_obj = Path(path)
        dir_name = path_obj.name
        
        # If directory name is empty or special directory, use parent directory name
        if not dir_name or dir_name in ['.', '..']:
            dir_name = path_obj.parent.name
            
        # Clean directory name, remove possible special characters
        import re
        clean_name = re.sub(r'[^\w\-_]', '_', dir_name)
        
        return clean_name if clean_name else "sequence"

    def get_default_output_path(self):
        """Generate default output path"""
        if self.video_name:
            return f"./processed_data/{self.video_name}"
        else:
            return "./processed_data/sequence"

    def sample_frames_uniformly(self, frame_paths, target_count):
        """Uniformly sample frames"""
        if target_count >= len(frame_paths):
            return frame_paths
        
        total_frames = len(frame_paths)
        interval = total_frames / target_count
        
        sampled_indices = []
        for i in range(target_count):
            index = min(int(i * interval), total_frames - 1)
            sampled_indices.append(index)
        
        return [frame_paths[i] for i in sampled_indices]

    def resize_and_crop_frame(self, frame, target_width, target_height):
        """Resize and crop frame, following resize_video.py logic"""
        if target_width <= 0 or target_height <= 0:
            return frame
            
        h, w = frame.shape[:2]
        
        # If already target size, return directly
        if w == target_width and h == target_height:
            return frame
        
        # Determine if landscape or portrait
        is_landscape = w >= h
        target_is_landscape = target_width >= target_height
        
        # Calculate scaling and cropping
        if is_landscape == target_is_landscape:
            # Same orientation
            if w / h > target_width / target_height:
                # Width ratio is larger, adjust height first
                new_height = target_height
                new_width = int(w * (target_height / h))
                frame = cv2.resize(frame, (new_width, new_height))
                # Crop center part
                start_x = (new_width - target_width) // 2
                frame = frame[:, start_x:start_x + target_width]
            else:
                # Height ratio is larger, adjust width first
                new_width = target_width
                new_height = int(h * (target_width / w))
                frame = cv2.resize(frame, (new_width, new_height))
                # Crop center part
                start_y = (new_height - target_height) // 2
                frame = frame[start_y:start_y + target_height, :]
        else:
            # Different orientation, need rotation or special handling
            # Simple handling here: scale and crop according to target ratio
            scale_w = target_width / w
            scale_h = target_height / h
            scale = max(scale_w, scale_h)  # Choose larger scale ratio to ensure coverage
            
            new_width = int(w * scale)
            new_height = int(h * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Center crop
            start_x = max(0, (new_width - target_width) // 2)
            start_y = max(0, (new_height - target_height) // 2)
            end_x = min(new_width, start_x + target_width)
            end_y = min(new_height, start_y + target_height)
            
            frame = frame[start_y:end_y, start_x:end_x]
            
            # If cropped size is insufficient, add padding (black borders)
            if frame.shape[:2] != (target_height, target_width):
                padded_frame = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
                h_offset = (target_height - frame.shape[0]) // 2
                w_offset = (target_width - frame.shape[1]) // 2
                padded_frame[h_offset:h_offset+frame.shape[0], w_offset:w_offset+frame.shape[1]] = frame
                frame = padded_frame
        
        return frame

    def process_frames(self, frame_paths, target_frames, target_width, target_height, orig_frames, orig_width, orig_height):
        """Process frame sequence: frame sampling and resolution adjustment"""
        # 1. Frame sampling
        if target_frames != orig_frames:
            frame_paths = self.sample_frames_uniformly(frame_paths, target_frames)
            guru.info(f"Frame count adjusted: {orig_frames} -> {len(frame_paths)}")
        
        # 2. Process resolution
        processed_paths = []
        need_resize = target_width != orig_width or target_height != orig_height
        
        if need_resize:
            guru.info(f"Resolution adjusted: {orig_width}x{orig_height} -> {target_width}x{target_height}")
            
                        # Create temporary directory to store adjusted frames
            temp_dir = os.path.join("./tmp", f"sam2_resized_{int(time.time())}")
            os.makedirs(temp_dir, exist_ok=True)
            
            for i, frame_path in enumerate(frame_paths):
                frame = iio.imread(frame_path)
                resized_frame = self.resize_and_crop_frame(frame, target_width, target_height)
                
                new_path = os.path.join(temp_dir, f"frame_{i:05d}.jpg")
                iio.imwrite(new_path, resized_frame)
                processed_paths.append(new_path)
            
            return processed_paths
        
        return frame_paths

    def init_sam_model(self):
        if self.sam_model is None:
            # Check if checkpoint file exists
            if not os.path.exists(self.checkpoint_dir):
                error_msg = (
                    f"Model checkpoint file does not exist: {self.checkpoint_dir}\n\n"
                    "Please download SAM2 model from:\n"
                    "🔗 Official repository: https://github.com/facebookresearch/sam2\n\n"
                    "Recommended model files:\n"
                    "• sam2_hiera_large.pt (Large model, recommended)\n"
                    "• sam2_hiera_base_plus.pt (Medium scale)\n"
                    "• sam2_hiera_small.pt (Small scale, fast)\n"
                    "• sam2_hiera_tiny.pt (Smallest scale, fastest)\n\n"
                    "Download links:\n"
                    "• Large: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt\n"
                    "• Base+: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt\n"
                    "• Small: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt\n"
                    "• Tiny: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt\n\n"
                    f"Please place the downloaded model file to: {self.checkpoint_dir}"
                )
                guru.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir)
            guru.info(f"loaded model checkpoint {self.checkpoint_dir}")


    def clear_points(self) -> tuple[None, None, None, str]:
        self.selected_points.clear()
        self.selected_labels.clear()
        message = "All points cleared, please select new points to update mask"
        return None, None, None, message





    def _clear_image(self):
        """
        clears image and all masks/logits for that image
        """
        self.image = None
        self.frame_index = 0
        self.cur_mask = None
        self.cur_logit = None
        self.masks_all = []
        self.bbox_masks_all = []

    def reset(self):
        self._clear_image()
        self.sam_model.reset_state(self.inference_state)

    def set_img_dir(self, img_dir: str) -> int:
        self._clear_image()
        self.img_dir = img_dir
        
        # Add debug information
        guru.info(f"Scanning directory: {img_dir}")
        if not os.path.exists(img_dir):
            guru.error(f"Directory does not exist: {img_dir}")
            return 0
            
        all_files = os.listdir(img_dir)
        guru.info(f"Total {len(all_files)} files in directory: {all_files}")
        
        self.img_paths = [
            os.path.abspath(os.path.join(img_dir, p)) for p in sorted(all_files) if isimage(p)
        ]
        
        guru.info(f"Found {len(self.img_paths)} image files: {[os.path.basename(p) for p in self.img_paths]}")
        
        # Extract video name
        self.video_name = self.extract_video_name(img_dir)
        
        return len(self.img_paths)

    def set_input_image(self, i: int = 0) -> np.ndarray | None:
        guru.debug(f"Setting frame {i} / {len(self.img_paths)}")
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self.frame_index = i
        image = iio.imread(self.img_paths[i])
        self.image = image

        return image

    def get_sam_features(self) -> tuple[str, np.ndarray | None]:
        try:
            guru.info(f"Starting SAM state initialization, directory: {self.img_dir}")
            guru.info(f"Number of image files in directory: {len(self.img_paths)}")
            self.inference_state = self.sam_model.init_state(video_path=self.img_dir)
            self.sam_model.reset_state(self.inference_state)
            msg = (
                "SAM feature extraction completed. "
                "Click points on the image to update mask, then submit to start tracking"
            )
            guru.info("SAM feature extraction successful")
            return msg, self.image
        except Exception as e:
            error_msg = f"SAM feature extraction failed: {str(e)}"
            guru.error(error_msg)
            return error_msg, self.image

    def set_positive(self) -> str:
        self.cur_label_val = 1.0
        return "Selecting positive points. Submit mask to start tracking"

    def set_negative(self) -> str:
        self.cur_label_val = 0.0
        return "Selecting negative points. Submit mask to start tracking"

    def add_point(self, frame_idx, i, j):
        """
        get the binary mask of the object
        """
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        # get the mask and logit
        mask, logit = self.get_sam_mask(
            frame_idx, np.array(self.selected_points, dtype=np.float32), np.array(self.selected_labels, dtype=np.int32)
        )
        self.cur_mask = mask
        self.cur_logit = logit

        return mask
    

    def get_sam_mask(self, frame_idx, input_points, input_labels):
        """
        :param frame_idx int
        :param input_points (np array) (N, 2)
        :param input_labels (np array) (N,)
        return (H, W) binary mask, (H, W) logits
        """
        assert self.sam_model is not None
        

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=0,  # use single mask with id 0
                points=input_points,
                labels=input_labels,
            )

        # return the first (and only) mask as binary mask
        mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()
        logit = out_mask_logits[0].squeeze().cpu().numpy()
        return mask, logit

    def create_bbox_mask(self, binary_mask, min_area=100, expand_ratio=0.1):
        """
        Create bounding box mask from binary mask
        
        Args:
            binary_mask: Binary mask (numpy array, bool or 0/1)
            min_area: Minimum bounding box area, boxes smaller than this will be ignored
            expand_ratio: Bounding box expansion ratio, e.g. 0.1 means expand by 10%
            
        Returns:
            bbox_mask: Bounding box mask image (numpy array, 0-255)
        """
        # Ensure input is in 0-255 format
        if binary_mask.dtype == bool:
            mask_img = (binary_mask * 255).astype(np.uint8)
        else:
            mask_img = (binary_mask * 255).astype(np.uint8)
        
        # Get image dimensions
        height, width = mask_img.shape
        
        # Create a new all-black image
        bbox_mask = np.zeros_like(mask_img)
        
        # Find all contours
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return bbox_mask
        
        # Process each contour
        for contour in contours:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate bounding box area
            area = w * h
            
            # If bounding box is too small, ignore it
            if area < min_area:
                continue
            
            # Calculate expanded bounding box
            expand_w = int(w * expand_ratio)
            expand_h = int(h * expand_ratio)
            
            # Calculate new bounding box coordinates, ensure not exceeding image boundaries
            new_x = max(0, x - expand_w // 2)
            new_y = max(0, y - expand_h // 2)
            new_w = min(width - new_x, w + expand_w)
            new_h = min(height - new_y, h + expand_h)
            
            # Set expanded bounding box region to white (255) on the all-black mask image
            bbox_mask[new_y:new_y+new_h, new_x:new_x+new_w] = 255
        
        return bbox_mask

    def apply_grayscale_to_mask_region(self, input_image_path, mask_array):
        """Convert white (255) mask regions to gray (128) in input image"""
        # Read input image
        input_img = Image.open(input_image_path).convert('RGB')
        
        # Convert to numpy array
        input_array = np.array(input_img)
        
        # Create copy of output array
        output_array = input_array.copy()
        
        # Find white pixel positions in mask
        white_pixels = mask_array > 200
        
        # Set these positions to gray (128) in input image
        output_array[white_pixels] = 128
        
        return output_array

    def create_mask_frame(self, template_image_array):
        """Create all-black mask frame"""
        height, width = template_image_array.shape[:2]
        return np.zeros((height, width), dtype=np.uint8)

    def generate_training_sequence(self, output_dir):
        """Generate training sequence, following batch_step2_onlyall.py logic"""
        if not self.masks_all or not self.bbox_masks_all:
            return None, None
            
        # Create temporary directory
        temp_dir = os.path.join(output_dir, 'temp_sequence')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Read all images
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
        
        frame_index = 0
        
        # Stage 0: Place first frame separately
        first_image = images[0]
        iio.imwrite(os.path.join(temp_dir, f'frame_{frame_index:03d}.jpg'), first_image)
        frame_index += 1
        
        # Stage 1: Copy first frame, then apply gray processing to all other frames
        # Copy first frame
        iio.imwrite(os.path.join(temp_dir, f'frame_{frame_index:03d}.jpg'), first_image)
        frame_index += 1
        
        # Process remaining frames (apply gray processing)
        for i, (img, bbox_mask) in enumerate(zip(images[1:], self.bbox_masks_all[1:]), 1):
            grayed_img = self.apply_grayscale_to_mask_region(self.img_paths[i], bbox_mask)
            iio.imwrite(os.path.join(temp_dir, f'frame_{frame_index:03d}.jpg'), grayed_img)
            frame_index += 1
        
        # Stage 2: Directly copy all original frames
        for img in images:
            iio.imwrite(os.path.join(temp_dir, f'frame_{frame_index:03d}.jpg'), img)
            frame_index += 1
        
        # Stage 3: Add mask frames (first frame is all black, other frames use corresponding bbox mask)
        # First frame is all black
        black_mask = self.create_mask_frame(first_image)
        iio.imwrite(os.path.join(temp_dir, f'frame_{frame_index:03d}.jpg'), black_mask)
        frame_index += 1
        
        # Remaining frames use bbox mask
        for bbox_mask in self.bbox_masks_all[1:]:
            iio.imwrite(os.path.join(temp_dir, f'frame_{frame_index:03d}.jpg'), bbox_mask)
            frame_index += 1
        
        # Generate training video
        train_dir = os.path.join(output_dir, 'traindata')
        os.makedirs(train_dir, exist_ok=True)
        
        total_frames = len(images)
        video_name = f'sequence_all_frames_{total_frames}.mp4'
        output_video = os.path.join(train_dir, video_name)
        
        # Use ffmpeg to generate mp4
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # Overwrite existing files
            '-framerate', '5',  # Set frame rate to 5fps
            '-i', os.path.join(temp_dir, 'frame_%03d.jpg'),
            '-c:v', 'libx264',  # Use h264 encoding
            '-pix_fmt', 'yuv420p',  # Set pixel format
            '-crf', '11',  # Set video quality
            output_video
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            guru.info(f"Training video generated: {output_video}")
        except subprocess.CalledProcessError as e:
            guru.error(f"Failed to generate training video: {e}")
            return None, None
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        return video_name, train_dir

    def create_inference_videos(self, output_dir):
        """Create inference videos"""
        if not self.masks_all or not self.bbox_masks_all:
            return None, None
            
        # Create temporary directories
        temp_rgb_dir = os.path.join(output_dir, 'temp_rgb')
        temp_mask_dir = os.path.join(output_dir, 'temp_mask')
        os.makedirs(temp_rgb_dir, exist_ok=True)
        os.makedirs(temp_mask_dir, exist_ok=True)
        
        # Save gray-processed frames (instead of original frames)
        for i, (img_path, bbox_mask) in enumerate(zip(self.img_paths, self.bbox_masks_all)):
            # Apply gray processing to each frame
            grayed_img = self.apply_grayscale_to_mask_region(img_path, bbox_mask)
            iio.imwrite(os.path.join(temp_rgb_dir, f'frame_{i:04d}.jpg'), grayed_img)
        
        # Save bbox mask frames
        for i, bbox_mask in enumerate(self.bbox_masks_all):
            iio.imwrite(os.path.join(temp_mask_dir, f'frame_{i:04d}.jpg'), bbox_mask)
        
        # Generate RGB video
        rgb_video_path = os.path.join(output_dir, 'inference_rgb.mp4')
        ffmpeg_cmd_rgb = [
            'ffmpeg', '-y',
            '-framerate', '25',
            '-i', os.path.join(temp_rgb_dir, 'frame_%04d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '11',
            rgb_video_path
        ]
        
        # Generate Mask video
        mask_video_path = os.path.join(output_dir, 'inference_mask.mp4')
        ffmpeg_cmd_mask = [
            'ffmpeg', '-y',
            '-framerate', '25',
            '-i', os.path.join(temp_mask_dir, 'frame_%04d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '11',
            mask_video_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd_rgb, check=True)
            subprocess.run(ffmpeg_cmd_mask, check=True)
            guru.info(f"Inference videos generated: {rgb_video_path}, {mask_video_path}")
        except subprocess.CalledProcessError as e:
            guru.error(f"Failed to generate inference videos: {e}")
            return None, None
        
        # Clean up temporary directories
        shutil.rmtree(temp_rgb_dir)
        shutil.rmtree(temp_mask_dir)
        
        return rgb_video_path, mask_video_path

    def run_tracker(self) -> tuple[str, str]:

        # Read images and drop the alpha channel
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
        
        self.masks_all = []  # Store binary masks for all frames
        self.bbox_masks_all = []  # Store bbox masks for all frames
        
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(self.inference_state, start_frame_idx=0):
                # Get the first (and only) mask as binary mask
                mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()
                self.masks_all.append(mask)
                
                # Create bbox mask from binary mask
                bbox_mask = self.create_bbox_mask(mask, min_area=50, expand_ratio=0.1)
                self.bbox_masks_all.append(bbox_mask)

        # Create visualization video with original masks (green)
        out_frames = []
        bbox_frames = []
        
        for img, mask, bbox_mask in zip(images, self.masks_all, self.bbox_masks_all):
            # Original mask visualization (green for mask)
            colored_mask = np.zeros_like(img)
            colored_mask[mask] = [0, 255, 0]  # Green color for mask
            out_frame = compose_img_mask(img, colored_mask, 0.5)
            out_frames.append(out_frame)
            
            # Bbox mask visualization (red for bbox)
            colored_bbox_mask = np.zeros_like(img)
            colored_bbox_mask[bbox_mask > 0] = [255, 0, 0]  # Red color for bbox
            bbox_frame = compose_img_mask(img, colored_bbox_mask, 0.3)
            bbox_frames.append(bbox_frame)
            
        # Save both videos
        tmp_dir = "./tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        out_vidpath = os.path.join(tmp_dir, "tracked_masks.mp4")
        bbox_vidpath = os.path.join(tmp_dir, "tracked_bbox_masks.mp4")
        iio.mimwrite(out_vidpath, out_frames)
        iio.mimwrite(bbox_vidpath, bbox_frames)
        
        message = f"Tracking complete! Original mask video: {out_vidpath}, Bounding box mask video: {bbox_vidpath}."
        instruct = "If the results are satisfactory, please process and save preprocessed data to the output directory!"
        return out_vidpath, bbox_vidpath, f"{message} {instruct}"

    def save_masks_to_dir(self, output_dir: str, concept_prefix: str = "", ckpt_path: str = "",
                         learning_rate: float = 1e-3, save_every_n_epochs: int = 50, 
                         epochs: int = 100, precision: str = "bf16") -> str:
        assert self.bbox_masks_all is not None and len(self.bbox_masks_all) > 0
        
        return self.save_processed_data(output_dir, concept_prefix, ckpt_path, 
                                      learning_rate, save_every_n_epochs, epochs, precision)
    
    def create_configs(self, output_dir: str, ckpt_path: str, learning_rate: float = 1e-3, 
                      save_every_n_epochs: int = 50, epochs: int = 100, precision: str = "bf16"):
        """Create training configuration files"""
        precision = normalize_training_precision(precision)
        configs_dir = os.path.join(output_dir, 'configs')
        os.makedirs(configs_dir, exist_ok=True)
        
        # Create dataset.toml
        dataset_config = f"""enable_ar_bucket = false

[[directory]]
path = '{os.path.join(output_dir, "traindata").replace(os.sep, "/")}'
num_repeats = 1
"""
        
        dataset_path = os.path.join(configs_dir, 'dataset.toml')
        with open(dataset_path, 'w', encoding='utf-8') as f:
            f.write(dataset_config)
        
        # Configure different parameters based on precision
        if precision == "bf16":
            transformer_dtype = 'bfloat16'
            optimizer_type = 'AdamW'
            stabilize_line = ""  # Don't add stabilize parameter
        else:  # fp8
            transformer_dtype = 'float8'
            optimizer_type = 'AdamW8bitKahan'
            stabilize_line = "stabilize = true"
        
        # Create training.toml
        training_config = f"""output_dir = '{os.path.join(output_dir, "lora").replace(os.sep, "/")}'
dataset = '{dataset_path.replace(os.sep, "/")}'

epochs = {epochs}
micro_batch_size_per_gpu = 1
pipeline_stages = 1
gradient_accumulation_steps = 1
gradient_clipping = 1
warmup_steps = 0

eval_every_n_epochs = 1000000
eval_before_first_step = true
eval_micro_batch_size_per_gpu = 1
eval_gradient_accumulation_steps = 1

save_every_n_epochs = {save_every_n_epochs}
checkpoint_every_n_minutes = 1000000000
activation_checkpointing = 'unsloth'
partition_method = 'parameters'
save_dtype = 'bfloat16'
caching_batch_size = 1
steps_per_print = 1
video_clip_mode = 'single_beginning'
blocks_to_swap = 32

[model]
type = 'wan'
ckpt_path = '{ckpt_path.replace(os.sep, "/")}'
dtype = 'bfloat16'
transformer_dtype = '{transformer_dtype}'
timestep_sample_method = 'uniform'

[adapter]
type = 'lora'
rank = 16
dtype = 'bfloat16'
exclude_linear_modules = ["k_img", "v_img"]

[optimizer]
type = '{optimizer_type}'
lr = {learning_rate}
betas = [0.9, 0.99]
weight_decay = 0.01
{stabilize_line}
"""
        
        training_path = os.path.join(configs_dir, 'training.toml')
        with open(training_path, 'w', encoding='utf-8') as f:
            f.write(training_config)
        
        return configs_dir, dataset_path, training_path

    def save_processed_data(self, output_dir: str, concept_prefix: str = "", ckpt_path: str = "", 
                           learning_rate: float = 1e-3, save_every_n_epochs: int = 50, 
                           epochs: int = 100, precision: str = "bf16") -> str:
        """Complete data processing and saving"""
        precision = normalize_training_precision(precision)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. Save original frames to source_frames directory
            source_frames_dir = os.path.join(output_dir, 'source_frames')
            os.makedirs(source_frames_dir, exist_ok=True)
            
            guru.info("Saving original frames to source_frames directory...")
            for i, img_path in enumerate(self.img_paths):
                # Read image and convert to RGB
                img = Image.open(img_path).convert('RGB')
                # Save as PNG
                output_path = os.path.join(source_frames_dir, f'{i:05d}.png')
                img.save(output_path)
            guru.info(f"Original frames saved to: {source_frames_dir}")
            
            # 2. Save bbox masks to source_masks directory
            source_masks_dir = os.path.join(output_dir, 'source_masks')
            os.makedirs(source_masks_dir, exist_ok=True)
            
            guru.info("Saving bbox masks to source_masks directory...")
            for i, bbox_mask in enumerate(self.bbox_masks_all):
                # bbox_mask is already in 0-255 format, no need to convert
                output_path = os.path.join(source_masks_dir, f'{i:05d}.png')
                Image.fromarray(bbox_mask).save(output_path)
            guru.info(f"Original masks saved to: {source_masks_dir}")
            
            # 3. Save concept prefix to prefix.txt
            prefix_file_path = os.path.join(output_dir, 'prefix.txt')
            with open(prefix_file_path, 'w', encoding='utf-8') as f:
                f.write(concept_prefix)
            guru.info(f"Concept prefix saved to: {prefix_file_path}")
            
            # 4. Create additional_edited_frames directory
            additional_frames_dir = os.path.join(output_dir, 'additional_edited_frames')
            os.makedirs(additional_frames_dir, exist_ok=True)
            guru.info(f"Created additional_edited_frames directory: {additional_frames_dir}")
            
            # 5. Generate training sequence
            guru.info("Generating training sequence...")
            video_name, train_dir = self.generate_training_sequence(output_dir)
            
            if video_name and train_dir:
                # 6. Generate txt file with same name as mp4 (using Florence model to generate caption)
                txt_filename = video_name.replace('.mp4', '.txt')
                txt_path = os.path.join(output_dir, 'traindata', txt_filename)
                
                # Generate caption for first frame
                first_frame_path = self.img_paths[0]
                caption_text = generate_caption(first_frame_path, concept_prefix=concept_prefix)
                
                # Write caption directly to txt file
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(caption_text)
                
                guru.info(f"Training data saved to: {train_dir}")
                guru.info(f"Text file: {txt_path}")
                guru.info(f"Generated caption: {caption_text[:80]}..." if len(caption_text) > 80 else f"Generated caption: {caption_text}")
            
            # 7. Generate inference videos
            guru.info("Generating inference videos...")
            rgb_video, mask_video = self.create_inference_videos(output_dir)
            
            # 8. Create training configuration files
            if ckpt_path:
                guru.info("Generating training configuration files...")
                configs_dir, dataset_config_path, training_config_path = self.create_configs(
                    output_dir, ckpt_path, learning_rate, save_every_n_epochs, epochs, precision
                )
                guru.info(f"Configuration files generated: {configs_dir}")
            
            # Generate final report
            total_frames = len(self.bbox_masks_all)
            message_parts = [
                f"Complete data processing finished! Processed {total_frames} frames in total.",
                f"📁 Source frames: {source_frames_dir}",
                f"📁 Source masks: {source_masks_dir}",
                f"📝 Concept prefix: {prefix_file_path}",
                f"📁 Additional edited frames: {additional_frames_dir}",
                f"📁 Training data: {train_dir}",
                f"🎬 Inference RGB video: {rgb_video}" if rgb_video else "",
                f"🎭 Inference mask video: {mask_video}" if mask_video else ""
            ]
            
            if ckpt_path:
                message_parts.extend([
                    f"⚙️ Configuration files directory: {configs_dir}",
                    f"📝 Dataset config: {os.path.basename(dataset_config_path)}",
                    f"🔧 Training config: {os.path.basename(training_config_path)}",
                    f"🔧 Training parameters: learning_rate={learning_rate}, epochs={epochs}, save_interval={save_every_n_epochs}, precision={precision}",
                    "",
                    "🚀 To start LoRA training, run the following command:",
                    f'NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config {training_config_path}',
                    "",
                    "🎬 After training completes, perform inference:",
                    f"1. Save the edited first frame (from {output_dir}/source_frames/00000.png) as: {output_dir}/edited_image.png (or .jpg)",
                    f"2. Run inference command: python inference.py --model_root_dir {ckpt_path} --data_dir {output_dir}",
                    "",
                    "🎯 For additional edited frames as reference:",
                    f"1. Put your edited frames (from {output_dir}/source_frames) to {output_dir}/additional_edited_frames",
                    f"2. Run: python predata_additional.py --data_dir {output_dir}",
                    f"3. Train additional LoRA: NCCL_P2P_DISABLE=\"1\" NCCL_IB_DISABLE=\"1\" deepspeed --num_gpus=1 train.py --deepspeed --config {output_dir}/configs/training_additional.toml",
                    f"4. Run inference with additional frames: python inference.py --model_root_dir {ckpt_path} --data_dir {output_dir} --additional"
                ])
            
            message = "\n".join([part for part in message_parts if part])
            guru.info(message)
            return message
            
        except Exception as e:
            error_msg = f"Error occurred during data processing: {str(e)}"
            guru.error(error_msg)
            return error_msg


def validate_frame_count(frames):
    """Validate if frame count follows 4N+1 format and is within 5-81 range"""
    if frames < 5 or frames > 81:
        return False, f"Frame count must be between 5-81, current value: {frames}"
    
    if (frames - 1) % 4 != 0:
        # Find closest valid value
        valid_values = [4*n + 1 for n in range(1, 21) if 5 <= 4*n + 1 <= 81]
        closest = min(valid_values, key=lambda x: abs(x - frames))
        return False, f"Frame count must follow 4N+1 format (N is positive integer), suggested: {closest}"
    
    return True, ""

def isimage(p):
    ext = os.path.splitext(p.lower())[-1]
    return ext in [".png", ".jpg", ".jpeg"]


def draw_points(img, points, labels):
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out








def compose_img_mask(img, color_mask, fac: float = 0.5):
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u





def make_demo(
    checkpoint_dir,
    model_cfg,
):
    prompts = PromptGUI(checkpoint_dir, model_cfg)
    
    # Preload Florence model - required dependency
    init_florence_model()

    start_instructions = (
        "Upload a video file to extract frames, or select a directory containing frame sequences."
    )
    
    with gr.Blocks() as demo:
        instruction = gr.Textbox(
            start_instructions, label="Instructions", interactive=False
        )
        
        with gr.Tab("Upload Video"):
            with gr.Column():
                input_video_field = gr.File(
                    label="Upload Video File", 
                    file_types=[".mp4", ".avi", ".mov", ".mkv"]
                )
                
                with gr.Row():
                    video_target_frames = gr.Number(49, label="Target Frame Count", 
                                                  info="Must be 4N+1 format (N is positive integer), range 5-81")
                    video_target_width = gr.Number(832, label="Target Width")
                    video_target_height = gr.Number(480, label="Target Height")
                    
                extract_button = gr.Button("Extract Frames")
                extracted_dir_field = gr.Text(
                    None, label="Extracted Frame Directory", interactive=False
                )
                
        with gr.Tab("Select Image Directory"):
            with gr.Row():
                img_dir_field = gr.Text(
                    None, label="Image Directory Path", placeholder="Enter directory path containing image frames"
                )
                load_dir_button = gr.Button("Load Directory")
                
            with gr.Row():
                target_frames = gr.Number(49, label="Target Frame Count", 
                                        info="Must be 4N+1 format (N is positive integer), range 5-81")
                target_width = gr.Number(832, label="Target Width")
                target_height = gr.Number(480, label="Target Height")

        frame_index = gr.Slider(
            label="Frame Index",
            minimum=0,
            maximum=0,
            value=0,
            step=1,
        )

        with gr.Row():
            with gr.Column():
                reset_button = gr.Button("Reset")
                input_image = gr.Image(
                    None,
                    label="Input Frame",
                    every=1,
                )
                with gr.Row():
                    pos_button = gr.Button("Positive Point Selection")
                    neg_button = gr.Button("Negative Point Selection")
                clear_button = gr.Button("Clear Points")

            with gr.Column():
                output_img = gr.Image(label="Current Selection")
                submit_button = gr.Button("Submit Mask for Tracking")
                with gr.Row():
                    final_video = gr.Video(label="Original Mask Video")
                    bbox_video = gr.Video(label="Bounding Box Mask Video")
                mask_dir_field = gr.Text(
                    "./processed_data/sequence", label="Data Processing Save Path", interactive=True
                )
                concept_prefix_field = gr.Text(
                    "p3rs0n,", label="Prompt Prefix", 
                    placeholder="p3rs0n,", 
                    info="Special word prefix added before generated descriptions, default value is fine, usually no need to modify"
                )
                ckpt_path_field = gr.Text(
                    "/home/user/LoRAEdit/Wan2.1-I2V-14B-480P", 
                    label="Model Checkpoint Path", 
                    placeholder="Enter complete path to model checkpoint",
                    info="Model checkpoint path used for training"
                )
                
                                # Training configuration options
                with gr.Row():
                    precision_field = gr.Dropdown(
                        choices=["bf16", "fp8"], 
                        value="bf16", 
                        label="Training Precision",
                        info="bf16: Better compatibility; fp8: Saves more VRAM but requires newer GPU"
                    )
                    epochs_field = gr.Number(
                        100, label="Training Epochs",
                        minimum=1, maximum=1000, step=1,
                        info="Recommended 50-300 epochs"
                    )
                
                with gr.Row():
                    learning_rate_field = gr.Number(
                        0.0001, label="Learning Rate", 
                        minimum=1e-6, maximum=1e-1, step=1e-5,
                        info="bf16 recommended 1e-4, fp8 recommended 1e-3"
                    )
                    save_every_n_epochs_field = gr.Number(
                        50, label="Save Interval (Epochs)", 
                        minimum=1, maximum=500, step=1,
                        info="Save model every N epochs"
                    )
                
                save_button = gr.Button("Process and Save Data")

        def load_image_directory(img_dir, target_frames, target_width, target_height):
            if not img_dir or not os.path.isdir(img_dir):
                error_msg = "Please enter a valid directory path"
                gr.Info(error_msg)
                return gr.Slider(), error_msg, None, "./processed_data/sequence"
            
            # Validate frame count
            is_valid, error_msg = validate_frame_count(int(target_frames))
            if not is_valid:
                gr.Warning(error_msg)
                return gr.Slider(), error_msg, None, "./processed_data/sequence"
                
            # First set original image directory
            num_imgs = prompts.set_img_dir(img_dir)
            if num_imgs == 0:
                error_msg = "No image files found in directory"
                gr.Warning(error_msg)
                return gr.Slider(), error_msg, None, "./processed_data/sequence"
            
            # Get original information
            first_frame = iio.imread(prompts.img_paths[0])
            orig_height, orig_width = first_frame.shape[:2]
            orig_frames = len(prompts.img_paths)
            
            # Process frame count and resolution
            processed_paths = prompts.process_frames(
                prompts.img_paths, 
                int(target_frames),
                int(target_width),
                int(target_height),
                orig_frames,
                orig_width,
                orig_height
            )
            
            # Update processed paths
            prompts.img_paths = processed_paths
            final_frames = len(processed_paths)
            
            slider = gr.Slider(minimum=0, maximum=final_frames - 1, value=0, step=1)
            first_image = prompts.set_input_image(0)
            
            # Automatically get SAM features
            sam_message, sam_image = prompts.get_sam_features()
            
            # Generate processing information
            process_info = []
            if target_frames != orig_frames:
                process_info.append(f"Frame count: {orig_frames} -> {final_frames}")
            if target_width != orig_width or target_height != orig_height:
                process_info.append(f"Resolution: {orig_width}x{orig_height} -> {target_width}x{target_height}")
            
            message = f"Loaded images from {img_dir}."
            if process_info:
                message += f" Processing: {', '.join(process_info)}."
            message += f" {sam_message}"
            
            # Show success message
            gr.Success(f"Successfully loaded {final_frames} frame images!")
            
            # Generate default output path
            default_output_path = prompts.get_default_output_path()
            
            return slider, message, sam_image if sam_image is not None else first_image, default_output_path

        def extract_frames_from_video(video_file, target_frames, target_width, target_height):
            if video_file is None:
                error_msg = "Please upload a video file first"
                gr.Info(error_msg)
                return error_msg, None, gr.Slider(), None, "./processed_data/sequence"
                
            try:
                # Get video information
                cap = cv2.VideoCapture(video_file)
                if not cap.isOpened():
                    error_msg = "Cannot open video file"
                    gr.Warning(error_msg)
                    return error_msg, None, gr.Slider(), None, "./processed_data/sequence"
                
                orig_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                cap.release()
                
                # Show success message
                success_msg = f"Video information retrieved successfully: {orig_frame_count} frames, {orig_width}x{orig_height}. Click 'Extract Frames' button to start processing."
                gr.Info(f"Video parsing successful! Original: {orig_frame_count} frames {orig_width}x{orig_height}")
                return success_msg, None, gr.Slider(), None, "./processed_data/sequence"
                
            except Exception as e:
                error_msg = f"Failed to get video information: {str(e)}"
                gr.Warning(error_msg)
                return error_msg, None, gr.Slider(), None, "./processed_data/sequence"
        
        def actually_extract_frames(video_file, target_frames, target_width, target_height):
            if video_file is None:
                error_msg = "Please upload a video file first"
                gr.Warning(error_msg)
                return error_msg, None, gr.Slider(), None, "./processed_data/sequence"
            
            # Validate frame count
            is_valid, error_msg = validate_frame_count(int(target_frames))
            if not is_valid:
                gr.Warning(error_msg)
                return error_msg, None, gr.Slider(), None, "./processed_data/sequence"
                
            # Create temporary directory to store extracted frames
            temp_dir = os.path.join("./tmp", f"video_frames_{int(time.time())}")
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                # Get video information
                cap = cv2.VideoCapture(video_file)
                if not cap.isOpened():
                    error_msg = "Cannot open video file"
                    gr.Warning(error_msg)
                    return error_msg, None, gr.Slider(), None, "./processed_data/sequence"
                
                orig_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Determine final frame count
                final_frames = int(target_frames)
                final_width = int(target_width)
                final_height = int(target_height)
                
                # Calculate sampling interval
                if final_frames >= orig_frame_count:
                    # Extract all frames
                    frame_indices = list(range(orig_frame_count))
                else:
                    # Uniform sampling
                    interval = orig_frame_count / final_frames
                    frame_indices = [min(int(i * interval), orig_frame_count - 1) for i in range(final_frames)]
                
                # Extract frames
                extracted_frames = 0
                guru.info(f"Starting frame extraction, target frame count: {len(frame_indices)}")
                guru.info(f"Save directory: {temp_dir}")
                
                for i, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        guru.warning(f"Cannot read frame {frame_idx}")
                        break
                    
                    # Adjust resolution
                    if orig_width != final_width or orig_height != final_height:
                        frame = prompts.resize_and_crop_frame(frame, final_width, final_height)
                    
                    # Save frame
                    output_path = os.path.join(temp_dir, f"{i:05d}.jpg")
                    try:
                        success = cv2.imwrite(output_path, frame)
                        if success:
                            extracted_frames += 1
                            if extracted_frames <= 3 or extracted_frames % 10 == 0:  # Only log first 3 and every 10th
                                guru.info(f"Successfully saved frame {i}: {output_path}")
                        else:
                            guru.error(f"cv2.imwrite returned failure: {output_path}")
                    except Exception as e:
                        guru.error(f"Exception occurred while saving frame {i}: {e}")
                
                guru.info(f"Actually extracted {extracted_frames} frames")
                
                cap.release()
                
                # Check what files are actually in the directory
                actual_files = os.listdir(temp_dir)
                guru.info(f"After extraction, files in directory {temp_dir}: {actual_files}")
                
                if extracted_frames == 0:
                    error_msg = "Failed to extract any frames"
                    gr.Warning(error_msg)
                    return error_msg, None, gr.Slider(), None, "./processed_data/sequence"
                
                # Automatically load extracted frames and get SAM features
                # Convert to absolute path
                abs_temp_dir = os.path.abspath(temp_dir)
                guru.info(f"Using absolute path: {abs_temp_dir}")
                num_imgs = prompts.set_img_dir(abs_temp_dir)
                if num_imgs == 0:
                    error_msg = "No image files found in extracted directory"
                    gr.Warning(error_msg)
                    return error_msg, temp_dir, gr.Slider(), None, "./processed_data/sequence"
                    
                slider = gr.Slider(minimum=0, maximum=num_imgs - 1, value=0, step=1)
                first_image = prompts.set_input_image(0)
                
                # Automatically get SAM features
                sam_message, sam_image = prompts.get_sam_features()
                
                # Generate processing information
                process_info = []
                if target_frames != orig_frame_count:
                    process_info.append(f"Frame count: {orig_frame_count} -> {extracted_frames}")
                if target_width != orig_width or target_height != orig_height:
                    process_info.append(f"Resolution: {orig_width}x{orig_height} -> {final_width}x{final_height}")
                
                message = f"Video frames extracted to: {temp_dir}, total {extracted_frames} frames."
                if process_info:
                    message += f" Processing: {', '.join(process_info)}."
                message += f" {sam_message}"
                
                # Show success message
                gr.Success(f"Successfully extracted {extracted_frames} frames! SAM features ready.")
                
                # Generate default output path
                default_output_path = prompts.get_default_output_path()
                
                return message, temp_dir, slider, sam_image if sam_image is not None else first_image, default_output_path
                
            except Exception as e:
                error_msg = f"Frame extraction failed: {str(e)}"
                gr.Warning(error_msg)
                return error_msg, None, gr.Slider(), None, "./processed_data/sequence"

        def get_select_coords(frame_idx, img, evt: gr.SelectData):
            if img is None:
                return None
            i = evt.index[1]  # type: ignore
            j = evt.index[0]  # type: ignore
            binary_mask = prompts.add_point(frame_idx, i, j)
            guru.debug(f"{binary_mask.shape=}")
            
            # create a simple green overlay for the mask
            colored_mask = np.zeros_like(img)
            colored_mask[binary_mask] = [0, 255, 0]  # green color for mask
            out_u = compose_img_mask(img, colored_mask, 0.5)
            out = draw_points(out_u, prompts.selected_points, prompts.selected_labels)
            return out

        # Event bindings
        load_dir_button.click(
            load_image_directory,
            [img_dir_field, target_frames, target_width, target_height],
            [frame_index, instruction, input_image, mask_dir_field]
        )
        
        # Automatically display information when video is uploaded
        input_video_field.change(
            extract_frames_from_video,
            [input_video_field, video_target_frames, video_target_width, video_target_height],
            [instruction, extracted_dir_field, frame_index, input_image, mask_dir_field]
        )
        
        # Click extract button to perform actual extraction
        extract_button.click(
            actually_extract_frames,
            [input_video_field, video_target_frames, video_target_width, video_target_height],
            [instruction, extracted_dir_field, frame_index, input_image, mask_dir_field]
        )

        frame_index.change(prompts.set_input_image, [frame_index], [input_image])
        input_image.select(get_select_coords, [frame_index, input_image], [output_img])

        reset_button.click(prompts.reset)
        clear_button.click(
            prompts.clear_points, outputs=[output_img, final_video, bbox_video, instruction]
        )
        pos_button.click(prompts.set_positive, outputs=[instruction])
        neg_button.click(prompts.set_negative, outputs=[instruction])

        def run_tracker_with_message():
            result = prompts.run_tracker()
            if result[0] and result[1]:  # If videos were successfully returned
                gr.Success("Object tracking completed! Please check the generated videos.")
            return result

        def update_learning_rate_on_precision_change(precision):
            """Automatically adjust learning rate based on precision selection"""
            precision = normalize_training_precision(precision)
            if precision == "bf16":
                return 0.0001  # 1e-4
            else:  # fp8
                return 0.001   # 1e-3

        def save_data_with_message(output_dir, concept_prefix, ckpt_path, learning_rate, 
                                 save_every_n_epochs, epochs, precision):
            # 1. Check if model checkpoint path exists
            if not os.path.exists(ckpt_path):
                error_msg = f"Model checkpoint path does not exist: {ckpt_path}"
                gr.Warning(error_msg)
                return error_msg
            
            # 2. Show start processing message
            gr.Info("Starting data processing, please wait...")
            
            # 3. Execute data processing
            result = prompts.save_masks_to_dir(output_dir, concept_prefix, ckpt_path,
                                             learning_rate, int(save_every_n_epochs), 
                                             int(epochs), precision)
            if "complete" in result.lower() or "success" in result.lower() or "finished" in result.lower():
                gr.Success("Data processing and saving completed! Please follow the instructions at the top of the page for LoRA training.")
            elif "error" in result.lower() or "failed" in result.lower():
                gr.Warning(f"Save failed: {result}")
            else:
                gr.Info(result)
            return result

        submit_button.click(run_tracker_with_message, outputs=[final_video, bbox_video, instruction])
        
        # Bind precision change event to automatically adjust learning rate
        precision_field.change(
            update_learning_rate_on_precision_change, 
            inputs=[precision_field], 
            outputs=[learning_rate_field]
        )
        
        save_button.click(
            save_data_with_message, 
            inputs=[mask_dir_field, concept_prefix_field, ckpt_path_field, 
                   learning_rate_field, save_every_n_epochs_field, epochs_field, precision_field], 
            outputs=[instruction]
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--checkpoint_dir", type=str, default="models_sam/sam2_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml")
    args = parser.parse_args()

    # Check if model file exists at startup
    if not os.path.exists(args.checkpoint_dir):
        print(f"\n❌ Model checkpoint file does not exist: {args.checkpoint_dir}")
        print("\n📥 Please download SAM2 model from:")
        print("🔗 Official repository: https://github.com/facebookresearch/sam2")
        print("\nRecommended model files:")
        print("• sam2_hiera_large.pt (Large model, recommended)")
        print("• sam2_hiera_base_plus.pt (Medium scale)")
        print("• sam2_hiera_small.pt (Small scale, fast)")
        print("• sam2_hiera_tiny.pt (Smallest scale, fastest)")
        print("\n📎 Direct download links:")
        print("• Large: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
        print("• Base+: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt")
        print("• Small: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt")
        print("• Tiny: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt")
        print(f"\n💡 Please place the downloaded model file to: {args.checkpoint_dir}")
        print("or create the corresponding directory structure")
        print("\nProgram will exit...")
        exit(1)

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    demo = make_demo(
        args.checkpoint_dir,
        args.model_cfg,
    )
    demo.launch(server_port=args.port)
