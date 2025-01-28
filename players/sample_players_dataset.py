import subprocess
import random
import os
import json

def sample_random_frames_ffmpeg(video_path, num_samples, start_frame=1000, output_dir="sampled_frames"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get video information using ffprobe with JSON output
    probe_cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    # Parse the JSON output
    probe_output = json.loads(subprocess.check_output(probe_cmd).decode())
    
    # Get video stream info
    video_stream = next(s for s in probe_output['streams'] if s['codec_type'] == 'video')
    
    # Get duration and frame rate
    duration = float(probe_output['format']['duration'])
    
    # Parse frame rate which might be in rational format (e.g., "30000/1001")
    frame_rate = video_stream.get('r_frame_rate', '30/1')
    if '/' in frame_rate:
        num, den = map(float, frame_rate.split('/'))
        frame_rate = num / den
    else:
        frame_rate = float(frame_rate)
    
    # Calculate total frames
    total_frames = int(duration * frame_rate)
    print(f"Video duration: {duration:.2f}s")
    print(f"Frame rate: {frame_rate:.2f}fps")
    print(f"Estimated total frames: {total_frames}")
    
    # Generate random frame numbers
    end_frame = total_frames - 1
    if start_frame >= end_frame:
        raise ValueError(f"Start frame {start_frame} exceeds total frames {end_frame}")
    
    frames_to_sample = random.sample(range(start_frame, end_frame), min(num_samples, end_frame - start_frame))
    frames_to_sample.sort()
    
    for frame_number in frames_to_sample:
        output_path = os.path.join(output_dir, f"4_frame_{frame_number}.jpg")
        
        # Convert frame number to timestamp
        timestamp = frame_number / frame_rate
        
        # Use ffmpeg to extract the frame
        cmd = [
            'ffmpeg',
            '-ss', f"{timestamp:.3f}",  # Seeking by timestamp
            '-i', video_path,
            '-vsync', '0',
            '-frames:v', '1',
            '-q:v', '1',     # Highest quality
            '-filter:v', 'yadif',  # Deinterlacing filter
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"Extracted frame at {timestamp:.3f}s (frame {frame_number})")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting frame {frame_number}: {e}")
            continue
# Example usage
if __name__ == "__main__":
    video_path = "/Users/cosmincojocaru/Downloads/Liga.1.Sepsi.Sf.Gheorghe.vs.U.Cluj.20.12.2024.1080i.HDTV.MPA2.0.H.264-playTV/Liga.1.Sepsi.Sf.Gheorghe.vs.U.Cluj.20.12.2024.1080i.HDTV.MPA2.0.H.264-playTV.mkv"  # Replace with your video path
    num_samples = 100  # Number of random frames to sample
    sample_random_frames_ffmpeg(video_path, num_samples)