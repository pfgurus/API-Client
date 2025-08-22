from torchaudio.io import StreamWriter
import torch
import requests
import tempfile
#from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy import VideoFileClip, concatenate_videoclips

def save_av_clip(frames, audio, output_path, video_frame_rate=25, audio_sample_rate=16000):
        """Saves a video and audio clip with the corrected audio handling."""
        frames = frames.cpu()
        audio = audio.cpu()
        
        if frames.dtype != torch.uint8:
            frames = (frames.clamp(0, 255)).to(torch.uint8)

        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)

        writer = StreamWriter(output_path, format='mp4')

        writer.add_video_stream(
            frame_rate=video_frame_rate,
            width=frames.shape[3],
            height=frames.shape[2],
            encoder="libx264",
            encoder_option={"crf": "18"},
        )
        writer.add_audio_stream(
            sample_rate=audio_sample_rate,
            num_channels=1,
        )

        with writer.open():
            writer.write_video_chunk(0, frames)
            writer.write_audio_chunk(1, audio.unsqueeze(1))

def stitch_video_stream(stream, output_path, verbose=False):
    """Downloads video chunks from a stream and stitches them into a single video file."""
    chunk_paths = []
    # Use a temporary directory that cleans itself up automatically
    with tempfile.TemporaryDirectory() as temp_dir:
        if verbose:
            print("Starting to download video chunks...")

        for i, chunk_url in enumerate(stream):
            if verbose:
                print(f"Downloading chunk {i+1} from {chunk_url}...")
            
            try:
                response = requests.get(chunk_url, stream=True)
                response.raise_for_status()
                
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp4")
                with open(chunk_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                chunk_paths.append(chunk_path)
            except requests.exceptions.RequestException as e:
                print(f"Failed to download chunk {i+1}: {e}")
                continue # Skip to the next chunk

        if not chunk_paths:
            print("No chunks were downloaded. Aborting video creation.")
            return None

        if verbose:
            print("\nAll chunks downloaded. Concatenating into final video...")

        try:
            video_clips = [VideoFileClip(path) for path in chunk_paths]
            final_clip = concatenate_videoclips(video_clips)
            
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True
            )
            
            # Close the clips to release file handles
            for clip in video_clips:
                clip.close()

        except Exception as e:
            print(f"An error occurred during video stitching: {e}")
            return None

    if verbose:
        print(f"Video successfully saved to: {output_path}")
        
    return output_path
