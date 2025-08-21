# Casablanca API Client

A simple Python client for interacting with the Casablanca AI Model API.

---

# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Issues](#issues)

---

## Installation

Install the library directly from the private GitHub repository using `pip`. You will need a GitHub Personal Access Token with `repo` scope to authenticate.

```bash
pip install git+https://github.com/pfgurus/API-Client.git
````

## Usage

The library provides three main components to interact with the API.

### 1. `APIClient`

This is the main class for making predictions. It handles authentication and communication with the API endpoint.

Following input is possible when using the `predict` method:


- *image_path*: Path to your local image or a downloadable asset in common image formats.
- *audio_path*: Path to your local audio file or a downloadable asset in common audio formats, or an `.mp4` file with extractable audio.
- *model*: The ID or name of the model to use for prediction. Use the method below to see available models.
- *guidance_scale*: Controls the strength of model guidance; higher values increase guidance, which may result in more pronounced effects or potential artifacts.
- *output_format*: `mp4` returns a link to the output video; `chunks` returns the model's unformatted output for further handling.
- *verbose*: If set to true, helpful information will be provided alongside the output.

```python
from casablanca_api import APIClient

# Initialize the client with your secret API key
client = APIClient(api_key="your-secret-user-api-key")

# Get a URL to the generated .mp4 file
video_url = client.predict(
    image_path="/path/to/image.png", # or https://.../image.png
    audio_path="/path/to/audio.mp3", # or https://.../audio.mp3
)
print(f"Video URL: {video_url}")

# Or, get a Raw object containing the raw data
chunks = client.predict(
    image_path="/path/to/image.png", # or https://.../image.png
    audio_path="/path/to/audio.mp3", # or https://.../audio.mp3"
    output_format="chunks"
)
```

### Display available models

You can display information about available models using:

```python
client = APIClient(api_key="your-secret-user-api-key")
client.list_models(display=True)  # Prints available models and their details
```

### 2. `RawData`

A data-handling class that automatically loads the raw frames and audio when you request the `"chunks"` format.

```python
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

```python
from casablanca_api.utils import save_av_clip

# Save the video from the Raw object
save_av_clip(frames, audio, "my_video.mp4")
print("Video saved successfully!")
 ```

For the RawData class and utility functions to work, you will need the following libraries:

- torchaudio: 

```bash
pip install torchaudio
````
- torch
```bash
pip install torch
````

## Issues

If you receive a JSON response or encounter unexpected behavior, it may be due to conflicts in your Python environment or missing dependencies. To resolve such issues, it is best to create a clean environment and reinstall the library and its dependencies. Follow the steps below for either Conda or pip/venv:

### Using Conda

```bash
conda create -n casablanca-env python=3.11
conda activate casablanca-env
pip install git+https://github.com/pfgurus/API-Client.git
```

### Using venv and pip

```bash
python3 -m venv casablanca-env
source casablanca-env/bin/activate
pip install git+https://github.com/pfgurus/API-Client.git
```

After setting up the new environment and installing the required packages, retry your API calls.


