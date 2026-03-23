import cv2
import numpy as np 
import os 
import glob 
import re
from pathlib import Path 

class StereoVisionPipeline:
    def __init__(self, config):
       
        self.config = config 
        self.calib = None
        self.left_img = None 
        self.right_img = None 
        self.disparity_map = None 
        self.depth_map = None 
         
        # Validate dataset path
        self.dataset_path = Path(config['dataset_path'])
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Load calibration
        self._load_calibration()
        
        # Initialize stereo matcher
        self._init_stereo_matcher() 
    

    def _load_calibration(self):
        calib_file = self.dataset_path / 'calib.txt'
        if not calib_file.exists():
            raise FileNotFoundError(f"calib.txt not found in {self.dataset_path}")
        
        self.calib = {}
        with open(calib_file, 'r') as f: 
            for line in f:
                line = line.strip() # Uklanjam prazne prostore sa pocetka i kraja linije
                if not line or line.startswith('#'):
                    continue
                    
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1) 
                    key = key.strip() 
                    
                    # Handle camera matrices (e.g., cam0=[...])
                    if key in ['cam0', 'cam1']: 
                        numbers = re.findall(r'[\d.]+', value) 
                        if len(numbers) >= 9:
                            # Reshape to 3x3 matrix
                            self.calib[key] = np.array(numbers[:9], dtype=float).reshape(3, 3)
                    else:
                        # Simple numeric values
                        try:
                            self.calib[key] = float(value) 
                        except ValueError:
                            self.calib[key] = value
        
        # Extract essential parameters
        if 'cam0' in self.calib: # Ako je kalibraciona matrica za cam0 ucitana, izvucicu iz nje potrebne parametre
            self.focal_length = self.calib['cam0'][0, 0]  # fx in pixels
            self.cx = self.calib['cam0'][0, 2]            # Principal point x
            self.cy = self.calib['cam0'][1, 2]            # Principal point y
        
        self.baseline = self.calib.get('baseline', 0) / 1000.0  # Convert mm to meters
        self.doffs = self.calib.get('doffs', 0) # Disparity offset, used in depth calculation
        
        print(f"Loaded calibration: f={self.focal_length:.1f}px, baseline={self.baseline:.3f}m")
    
    def _init_stereo_matcher(self): # Initialize the stereo matching algorithm. 
        """Initialize OpenCV stereo matching algorithm"""
        method = self.config.get('stereo_method', 'SGBM').upper()
        num_disparities = self.config.get('num_disparities', 64)
        block_size = self.config.get('block_size', 11)
        
        # Ensure num_disparities is multiple of 16
        num_disparities = ((num_disparities + 15) // 16) * 16 # OpenCV zahteva da broj dispariteta bude deljiv sa 16, ovo osigurava da se algoritam pravilno izvrsava
        
        if method == 'BM': 
            # StereoBM - faster, less accurate
            self.stereo = cv2.StereoBM.create(
                numDisparities=num_disparities,
                blockSize=block_size
            )
            # Optional parameters
            self.stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
            self.stereo.setPreFilterSize(9)
            self.stereo.setPreFilterCap(31)
            self.stereo.setTextureThreshold(10)
            self.stereo.setUniquenessRatio(15)
            self.stereo.setSpeckleRange(32)
            self.stereo.setSpeckleWindowSize(100)
            
        else:  # Default to SGBM - more accurate, slower. 
            self.stereo = cv2.StereoSGBM.create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=block_size
            )
            # SGBM-specific parameters
            self.stereo.setP1(8 * 3 * block_size ** 2)
            self.stereo.setP2(32 * 3 * block_size ** 2)
            self.stereo.setDisp12MaxDiff(1)
            self.stereo.setUniquenessRatio(10)
            self.stereo.setSpeckleWindowSize(100)
            self.stereo.setSpeckleRange(32)
            self.stereo.setMode(cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        
        print(f"Initialized {method} with numDisparities={num_disparities}, blockSize={block_size}")
    
    def load_images(self, left_file='im0.png', right_file='im1.png'):
        left_path = self.dataset_path / left_file
        right_path = self.dataset_path / right_file
        
        if not left_path.exists():
            # Try alternative naming (im2.png, im6.png for older Middlebury)
            left_path = self.dataset_path / 'im2.png'
            right_path = self.dataset_path / 'im6.png'
        
        if not left_path.exists():
            raise FileNotFoundError(f"No left image found in {self.dataset_path}")
        
        # Load images
        self.left_img = cv2.imread(str(left_path))
        self.right_img = cv2.imread(str(right_path))
        
        if self.left_img is None or self.right_img is None:
            raise ValueError("Failed to load images")
        
        # Convert to grayscale if needed
        if self.config.get('use_gray', True):
            self.left_gray = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY)
            self.right_gray = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2GRAY)
        else:
            self.left_gray = self.left_img
            self.right_gray = self.right_img
        
        print(f"Loaded images: {left_path.name}, {right_path.name} ({self.left_img.shape[1]}x{self.left_img.shape[0]})")
    
    def compute_disparity(self): 
        if self.left_gray is None or self.right_gray is None:
            raise ValueError("Images not loaded. Call load_images() first.")
        
        # Compute disparity
        disparity = self.stereo.compute(self.left_gray, self.right_gray).astype(np.float32) # OpenCV stereo compute returns a disparity map where valid disparities are scaled by 16 (for subpixel accuracy), 
        
        # For StereoBM/SGBM, disparity is scaled by 16
        if self.config.get('stereo_method', 'SGBM').upper() in ['BM', 'SGBM']:
            disparity = disparity / 16.0
        
        # Filter out invalid disparities (negative values)
        self.disparity_map = np.maximum(disparity, 0) 
        
        # Optional: Apply median filter to reduce noise
        if self.config.get('apply_median_filter', True):
            self.disparity_map = cv2.medianBlur(self.disparity_map.astype(np.float32), 5) 
        
        print(f"Disparity computed: range [{self.disparity_map.min():.1f}, {self.disparity_map.max():.1f}]") 
        
        return self.disparity_map
    
    def compute_depth(self):

        if self.disparity_map is None:
            raise ValueError("Disparity map not computed. Call compute_disparity() first.")
        
        # Avoid division by zero and invalid disparities
        valid_disparity = self.disparity_map > 0
        
        # Initialize depth map with zeros
        self.depth_map = np.zeros_like(self.disparity_map)
        
        # Compute depth only for valid disparities
        with np.errstate(divide='ignore', invalid='ignore'): 
            self.depth_map[valid_disparity] = (
                self.focal_length * self.baseline
            ) / (self.disparity_map[valid_disparity] + self.doffs)
        
        # Clip to reasonable range (e.g., 0.1m to 100m)
        max_depth = self.config.get('max_depth_meters', 50.0)
        self.depth_map = np.clip(self.depth_map, 0, max_depth) 
        
        print(f"Depth computed: range [{self.depth_map[self.depth_map>0].min():.2f}m, {self.depth_map.max():.2f}m]")
        
        return self.depth_map
   
    def visualize_results(self, save_fig=False, output_dir='output'): 
        if self.left_img is None or self.disparity_map is None:
            raise ValueError("No results to visualize")

        # Create output directory if saving
        if save_fig:
            os.makedirs(output_dir, exist_ok=True)

        # Prepare disparity and depth for visualization
        disp_vis = self.disparity_map.copy()
        if disp_vis.max() > 0:
            disp_vis = (disp_vis / disp_vis.max() * 255).astype(np.uint8) 
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        depth_vis = self.depth_map.copy()
        if depth_vis.max() > 0:
            p95 = np.percentile(depth_vis[depth_vis > 0], 95)
            depth_vis = np.clip(depth_vis, 0, p95)
            depth_vis = (depth_vis / p95 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS) 

        # Original image size
        h, w = self.left_img.shape[:2]

        # Create blank info panel BEFORE resizing
        info_panel = np.zeros((h, w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(info_panel, f"Focal: {self.focal_length:.0f}px", (10, 30), font, 0.7, (255,255,255), 2)
        cv2.putText(info_panel, f"Baseline: {self.baseline*1000:.1f}mm", (10, 60), font, 0.7, (255,255,255), 2)
        cv2.putText(info_panel, f"Method: {self.config.get('stereo_method', 'SGBM')}", (10, 90), font, 0.7, (255,255,255), 2)
        cv2.putText(info_panel, f"Disparities: {self.config.get('num_disparities', 64)}", (10, 120), font, 0.7, (255,255,255), 2)

        # Get display scale
        scale = self.config.get('display_scale', 1.0)
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            left_resized = cv2.resize(self.left_img, (new_w, new_h))
            disp_resized = cv2.resize(disp_color, (new_w, new_h))
            depth_resized = cv2.resize(depth_color, (new_w, new_h))
            info_panel = cv2.resize(info_panel, (new_w, new_h))  # resize after initialization
            w, h = new_w, new_h
        else:
            left_resized = self.left_img
            disp_resized = disp_color
            depth_resized = depth_color

        # Combine into 2x2 grid
        top_row = np.hstack([left_resized, disp_resized])
        bottom_row = np.hstack([depth_resized, info_panel])
        visualization = np.vstack([top_row, bottom_row])

        # Add labels
        cv2.putText(visualization, "Left Image", (10, 30), font, 1, (255,255,255), 2)
        cv2.putText(visualization, "Disparity Map", (w+10, 30), font, 1, (255,255,255), 2)
        cv2.putText(visualization, "Depth Map", (10, h+30), font, 1, (255,255,255), 2)
        cv2.putText(visualization, "Parameters", (w+10, h+30), font, 1, (255,255,255), 2)

        # Display
        cv2.imshow("Stereo Vision Pipeline Results", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save if requested
        if save_fig:
            output_path = os.path.join(output_dir, f"result_{Path(self.dataset_path).name}.png")
            cv2.imwrite(output_path, visualization)
            print(f"Visualization saved to {output_path}") 
   
    def save_disparity_map(self, output_dir='output'):
        """Save disparity map as image and numpy array"""
        os.makedirs(output_dir, exist_ok=True)
        scene_name = Path(self.dataset_path).name
        
        # Save as normalized image
        disp_norm = (self.disparity_map / self.disparity_map.max() * 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(output_dir, f"{scene_name}_disp.png"), disp_norm)
        
        # Save as numpy array
        np.save(os.path.join(output_dir, f"{scene_name}_disp.npy"), self.disparity_map)
        
        # Save depth map
        np.save(os.path.join(output_dir, f"{scene_name}_depth.npy"), self.depth_map)
        
        print(f"Results saved to {output_dir}/")


def main():
    """
    Main function demonstrating pipeline usage
    """
    
    # Configuration
    config = {
        # Path to downloaded Middlebury scene
        'dataset_path': './all/data/curule1',  # Change this to your path
        
        # Stereo algorithm selection
        'stereo_method': 'SGBM',  # 'BM' for faster, 'SGBM' for more accurate
        
        # Algorithm parameters
        'num_disparities': 196,     # Multiple of 16, larger = farther range
        'block_size': 7,           # Odd number, 5-15 typically
        
        # Preprocessing
        'use_gray': True,           # Convert to grayscale
        'apply_median_filter': True, # Reduce noise in disparity
        
        # Depth range
        'max_depth_meters': 20,    # Clip depth beyond this
        
        # Output options
        'save_results': True,
        'output_dir': './stereo_output',
        'display_scale': 0.4  # Scale down for display (1.0 = original size)
    }
    
    try:
        # Initialize pipeline
        print("=" * 50)
        print("Stereo Vision Pipeline - Middlebury 2021 Dataset")
        print("=" * 50)
        
        pipeline = StereoVisionPipeline(config)
        
        # Load images
        pipeline.load_images()
        
        # Compute disparity
        print("\nComputing disparity map...")
        pipeline.compute_disparity()
        
        # Compute depth
        print("\nConverting to depth map...")
        pipeline.compute_depth()
        
        # Visualize
        print("\nDisplaying results...")
        pipeline.visualize_results(save_fig=config['save_results'], 
                                   output_dir=config['output_dir'])
        
        # Save if requested
        if config['save_results']:
            pipeline.save_disparity_map(output_dir=config['output_dir'])
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


def batch_process_scenes(scenes_folder, config): 
    
    scenes = glob.glob(os.path.join(scenes_folder, "*"))
    scenes = [s for s in scenes if os.path.isdir(s) and 
              os.path.exists(os.path.join(s, "calib.txt"))]
    
    print(f"Found {len(scenes)} scenes to process")
    
    results = {}
    for scene_path in scenes:
        scene_name = Path(scene_path).name
        print(f"\n--- Processing {scene_name} ---")
        
        # Update config for this scene
        scene_config = config.copy()
        scene_config['dataset_path'] = scene_path
        
        try:
            pipeline = StereoVisionPipeline(scene_config)
            pipeline.load_images()
            pipeline.compute_disparity()
            pipeline.compute_depth()
            
            # Store statistics
            results[scene_name] = {
                'disparity_range': (float(pipeline.disparity_map.min()), 
                                   float(pipeline.disparity_map.max())),
                'depth_range': (float(pipeline.depth_map[pipeline.depth_map>0].min()),
                               float(pipeline.depth_map.max()))
            }
            
            if scene_config.get('save_results', False):
                pipeline.save_disparity_map(
                    output_dir=os.path.join(scene_config['output_dir'], scene_name))
            
        except Exception as e:
            print(f"Error processing {scene_name}: {e}")
            results[scene_name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Run main example
    exit_code = main()
    
   
