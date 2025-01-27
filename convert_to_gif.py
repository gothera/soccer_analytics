from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(input_path, output_path, resize_factor=0.5):
    """
    Convert MP4 video to GIF with optional resizing to keep file size manageable.
    
    Args:
        input_path (str): Path to input MP4 file
        output_path (str): Path for output GIF file
        resize_factor (float): Factor to resize the video (0.5 = half size)
    """
    # Load the video clip
    video_clip = VideoFileClip(input_path)
    
    # Resize video to reduce final GIF size
    resized_clip = video_clip.resize(resize_factor)
    
    # Write GIF file
    resized_clip.write_gif(output_path, fps=15)
    
    # Close clips to free up system resources
    video_clip.close()
    resized_clip.close()

# Example usage
convert_mp4_to_gif("./tactical_camera.mp4", "tactical.gif")