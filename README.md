# Casablanca API Client

A simple Python client for interacting with the Casablanca AI Model API.

---

# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Issues](#issues)

---

## Installation

Install the library directly from the GitHub repository using `pip`. 

```bash
pip install git+https://github.com/pfgurus/API-Client.git
````

## Usage

The library provides three main components to interact with the API.

### 1. `APIClient`

This is the main class for making predictions. It handles authentication and communication with the API endpoint.

The client supports two main workflows depending on the client's `api_set` value and which arguments you pass:

- Audio+Image -> Video (default non-streaming or streaming models)
- Text-to-Video (when `api_set="text_to_video"`)

Common `APIClient.predict` parameters (high-level):

- `image_path` (str, optional): Path or URL to the source image used as visual input.
- `audio_path` (str, optional): Path or URL to the audio file used to drive animation and lip sync. Required for audio+image workflows.
- `text` (str, optional): Prompt text used when generating a video from text. Required for text-to-video workflows.
- `model` (str, optional): Model id/name to use. Examples: `default`, `atv_stream`, or any id returned by `list_models()`.
- `guidance_scale` (float, optional): Strength of guidance for visual generation. Higher values increase guidance intensity (may increase artifacts).
- `output_format` (str, optional): `mp4` (default) returns a single video URL; `chunks` returns a `RawData`-compatible value for downstream handling.
- `verbose` (bool, optional): If True, the client prints progress and diagnostic information.

Text-to-Video specific parameters (when using `api_set="text_to_video"`):

- `voice_id` (str, optional): Identifier for the TTS/voice to use. Default in the client: `"Deep_Voice_Man"`.
- `language_boost` (str or None): Boost factor to bias the TTS/model toward a particular language characteristic.
- `emotion` (str or None): Optional emotion/style hint for the generated speech or animation (model-dependent).

Return values

- For non-streaming models and `output_format="mp4"`: `predict()` returns a string URL to the generated MP4 video.
- For `output_format="chunks"`: `predict()` returns a `RawData` object that can be inspected and saved using utilities.
- For streaming models (e.g., `model="atv_stream"`) the client will yield chunk URLs as they become available (useful for stitching or progressive playback).

Examples

# 1) Simple image+audio -> mp4 (batch)
```python
from casablanca_api import APIClient

# Initialize the client with your secret API key (default api_set supports audio+image)
client = APIClient(api_key="your-secret-user-api-key", api_set="audio_to_video")

video_url = client.predict(
    image_path="/path/to/image.png",   # local path or https:// URL
    audio_path="/path/to/audio.mp3",   # local path or https:// URL
    model="default",
    guidance_scale=1.5,
    output_format="mp4",
    verbose=True,
)
print(f"Video URL: {video_url}")
```

# 2) Image+audio -> chunks (raw frames/audio for advanced handling)
```python
chunks = client.predict(
    image_path="/path/to/image.png",
    audio_path="/path/to/audio.mp3",
    model="atv_stream",
    output_format="chunks",
    verbose=True,
)
```

# 3) Text-to-Video: set `api_set='text_to_video'` when constructing the client
```python
client = APIClient(api_key="your-secret-user-api-key", api_set="text_to_video")

# Generate a talking video from an image and a text prompt
video_url = client.predict(
    image_path="/path/to/portrait.png",
    text="Hello! This is a demo of text-to-video using Casablanca.",
    model="default",
    voice_id="Deep_Voice_Man",      # optional
    language_boost="German",        # optional
    emotion="happy",                # optional
    verbose=True,
)
print(f"Text-to-video URL: {video_url}")
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
### 3. Utility Functions

#### `save_av_clip`


A utility function to save the data from a `RawData` object to a playable `.mp4` video file.

```python
from casablanca_api.utils import save_av_clip

# Save the video from the Raw object
save_av_clip(frames, audio, "my_video.mp4")
print("Video saved successfully!")
 ```

---

#### `stitch_video_stream`

This utility function downloads video chunks from a list of URLs (such as those returned by the API in streaming mode) and stitches them into a single playable `.mp4` file. It uses `moviepy` for concatenation and handles temporary storage and cleanup automatically.


- `stream`: List of URLs to video chunks (e.g., returned by the API in streaming mode)
- `output_path`: Path to save the final stitched video
- `verbose`: If True, prints progress information

**Example usage:**

```python
chunks = client.predict(
    image_path=LOCAL_IMAGE_PATH,
    audio_path=LOCAL_AUDIO_PATH,
    model="atv_stream",
    output_format="chunks",
    verbose=True
)

final_video_path = "final_output.mp4"
stitch_video_stream(chunks, final_video_path, verbose=True)
```

This will download all video chunks and concatenate them into `final_output.mp4`.

For the RawData class and utility functions to work, you will need the following libraries:

- torchaudio: 

```bash
pip install torchaudio
````
- torch
  
```bash
pip install torch
````
- moviepy

```bash
pip install moviepy
````

### Text-to-Video inputs

These parameters let you control the speech, language, and emotional style of text-driven videos. Use them to make the generated narration and facial animation match the character and tone you want.

What you can tweak

- Voice (voice_id): pick a persona for the TTS. Examples:
    - `Wise_Woman`, `Friendly_Person`, `Inspirational_Girl`, `Deep_Voice_Man`, `Calm_Woman`, `Casual_Guy`
    - `Lively_Girl`, `Patient_Man`, `Young_Knight`, `Determined_Man`, `Lovely_Girl`, `Decent_Boy`
    - `Imposing_Manner`, `Elegant_Man`, `Abbess`, `Sweet_Girl_2`, `Exuberant_Girl`

- Emotion (emotion): tone/style hints for the generated speech and animation. Common values: `auto`, `neutral`, `happy`, `sad`, `angry`, `fearful`, `disgusted`, `surprised`.

- Language boost (language_boost): bias the TTS/model toward a particular language or locale. Use `None` for automatic behavior or specify a language hint such as `English`, `Chinese`, `Spanish`, `French`, `Arabic`, `Russian`, `Portuguese`, `German`, `Japanese`, `Korean`, etc.

Quick usage tip

When creating the client for text-to-video, set `api_set="text_to_video"` and pass the extras to `predict()`:

```python
client = APIClient(api_key="...")
client.api_set = "text_to_video"

url = client.predict(
        image_path="/path/to/portrait.png",
        text="Hello — let's demo a friendly voice with a happy tone.",
        voice_id="Friendly_Person",
        language_boost="English",
        emotion="happy",
)
```

Notes and references

- For detailed TTS options and model-specific guidance refer to the speech model docs (for example: `speech-02-turbo`) and the associated research: https://replicate.com/minimax/speech-02-turbo and https://arxiv.org/pdf/2505.07916

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


