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
