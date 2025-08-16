import requests
import json
import base64
import mimetypes
import os
from .handle_raw import RawData

class APIClient:
    def __init__(self, api_key, vercel_api_url="https://atv-model-api.vercel.app/api/predict"):
        if not api_key:
            raise ValueError("An API key is required.")
        self.api_key = api_key
        self.vercel_api_url = vercel_api_url

    def _file_to_data_uri(self, file_path):
        """Reads a local file and converts it to a Base64 data URI."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            raise ValueError(f"Could not determine MIME type for file: {file_path}")

        with open(file_path, "rb") as file:
            encoded_data = base64.b64encode(file.read()).decode('utf-8')

        return f"data:{mime_type};base64,{encoded_data}"

    def predict(self, image_path, audio_path, guidance_scale=1.0, output_format="mp4"):
        """Calls the prediction API with local file paths."""
        try:
            image_uri = self._file_to_data_uri(image_path)
            audio_uri = self._file_to_data_uri(audio_path)

            input_data = {
                "source_image": image_uri,
                "audio": audio_uri,
                "guidance_scale": guidance_scale,
                "output_format": output_format
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            response = requests.post(
                self.vercel_api_url,
                headers=headers,
                json=input_data,
                timeout=600  
            )
            response.raise_for_status()
            
            output_url = response.json()

            if output_format == "chunks":
                return RawData(data_url=output_url)
            else:
                # For mp4, just return the URL string
                return output_url
        
        except (requests.exceptions.RequestException, FileNotFoundError, ValueError) as e:
            print(f"\nAn error occurred: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Server responded with status {e.response.status_code}:")
                try:
                    print(json.dumps(e.response.json(), indent=2))
                except json.JSONDecodeError:
                    print(e.response.text)
            return None