import torch
import os
import requests
import tempfile
from torchaudio.io import StreamWriter

class RawData:
    def __init__(self, data_url):
        """
        Initializes the Raw object by downloading and loading chunk data from a URL.
        """
        self.frames = None
        self.audio = None
        self._load_data_from_url(data_url)

    def _load_data_from_url(self, url):
        """Downloads a file, saves it with a .pt extension, loads it, then deletes it."""
        tmp_file_path = None # Ensure variable is defined
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                    for chunk in r.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                    tmp_file_path = tmp_file.name
            
            loaded_data = torch.load(tmp_file_path)
            
            self.frames = loaded_data.get('frames')
            self.audio = loaded_data.get('audio')
            
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def display_info(self):
        """Prints descriptive information about the loaded chunks."""
        if self.frames is None or self.audio is None:
            print("No data loaded.")
            return

        num_frames = self.frames.shape[0]
        frame_shape = self.frames.shape[1:]
        audio_length_samples = self.audio.shape[1]
        
        print("--- Raw Chunk Information ---")
        print(f"Video Frames: {num_frames}")
        print(f"Frame Shape (C, H, W): {frame_shape}")
        print(f"Audio Samples: {audio_length_samples}")
        print("---------------------------")

    def save_av_clip(self, frames, audio, output_path, video_frame_rate=25, audio_sample_rate=16000):
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