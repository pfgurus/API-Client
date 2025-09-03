
from casablanca_api.client import APIClient
import time

LOCAL_IMAGE_PATH = "/home/lorenz/Casablanca/Data/GazeData/dgaze-3.0-beta3/1/FFHQ/00000/00653.png"

client_call = APIClient(api_key="C%$_ASPXCVVR//2461a437664BXRERFSA12$FSERHAaFSSsdzed/TRCVG", api_set="text_to_video")

print("--- Calling Text-to-Video API ---")
video = client_call.predict(
    model="atv_sync",
    image_path=LOCAL_IMAGE_PATH,
    text="Hello world, I am an avatar generated using only text and an image. This generation is done through the Casablanca AI API using just a few lines of Python code.",
    verbose=True,
    voice_id="Friendly_Person",
    language_boost="English",
    emotion="auto"
)

print("Video URL:", video)