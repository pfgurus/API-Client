# Casablanca API Client

A simple Python client for interacting with the Casablanca AI Model API.

---

## Installation

Install the library directly from the private GitHub repository using `pip`. You will need a GitHub Personal Access Token with `repo` scope to authenticate.

```bash
pip install git+https://<YOUR-TOKEN>@github.com/pfgurus/API-Client.git
````

## Usage

The library provides three main components to interact with the API.

### 1. `APIClient`

This is the main class for making predictions. It handles authentication and communication with the API endpoint.

```
from casablanca_api import APIClient

# Initialize the client with your secret API key
client = APIClient(api_key="your-secret-user-api-key")

# Get a URL to the generated .mp4 file
video_url = client.predict(
    image_path="https://.../image.png",
    audio_path="https://.../audio.mp3"
)
print(f"Video URL: {video_url}")

# Or, get a Raw object containing the raw data
chunks = client.predict(
    image_path="https://.../image.png",
    audio_path="https://.../audio.mp3",
    output_format="chunks"
)
```

### 2. `RawData`

A data-handling class that automatically loads the raw frames and audio when you request the `"chunks"` format.

```
from casablanca_api import RawData

# 'chunks' is an instance of the RawData class
chunks.display_info()
# --- Raw Chunk Information ---
# Video Frames: 110
# Frame Shape (C, H, W): torch.Size([3, 256, 256])
# Audio Samples: 70400
# ---------------------------
```

### 3. `save_av_clip`

A utility function to save the data from a `RawData` object to a playable `.mp4` video file.

```
from casablanca_api.utils import save_av_clip

# Save the video from the Raw object
save_av_clip(frames, audio, "my_video.mp4")
print("Video saved successfully!")
```
