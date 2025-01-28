from moviepy.editor import VideoFileClip, ColorClip, concatenate_videoclips

def merge_videos_with_transition(video_paths, output_path, transition_duration=1.0):
    """
    Merge multiple MP4 videos with black frame transitions between them.
    
    Args:
        video_paths (list): List of paths to input video files
        output_path (str): Path where the output video will be saved
        transition_duration (float): Duration of black frame transition in seconds
    """
    # Load all video clips
    video_clips = [VideoFileClip(path) for path in video_paths]
    
    # Create a black frame clip for transition
    # Using the first video's dimensions for the black frame
    width = video_clips[0].w
    height = video_clips[0].h
    black_clip = ColorClip(size=(width, height), 
                          color=(0, 0, 0), 
                          duration=transition_duration)
    
    # Create final clip list with black frames between videos
    final_clips = []
    for i, clip in enumerate(video_clips):
        final_clips.append(clip)
        # Add black transition after each video except the last one
        if i < len(video_clips) - 1:
            final_clips.append(black_clip)
    
    # Concatenate all clips
    final_video = concatenate_videoclips(final_clips)
    
    # Write the output video
    final_video.write_videofile(output_path, 
                              codec='libx264',
                              audio_codec='aac')
    
    # Close all clips to free up resources
    for clip in video_clips:
        clip.close()
    final_video.close()

# Example usage
if __name__ == "__main__":
    # List of input video paths
    videos = [
        "/Users/cosmincojocaru/playground/output_nb1_ball_1.mp4",
        "/Users/cosmincojocaru/playground/output_nb1_ball_2.mp4",
        "/Users/cosmincojocaru/playground/output_nb1_ball_3.mp4"
    ]
    
    # Output video path
    output = "merged_output.mp4"
    
    # Merge videos with 1-second black transitions
    merge_videos_with_transition(videos, output, transition_duration=1.0)