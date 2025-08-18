import requests
import json
import base64
import mimetypes
import os
import os
import base64
import mimetypes
import requests
import time
import json
from .handle_raw import RawData

class APIClient:
    def __init__(self, api_key, base_vercel_url="https://atv-model-api.vercel.app"):
        if not api_key:
            raise ValueError("An API key is required.")
        self.api_key = api_key
        # The base URL for the Vercel App.
        self.base_url = base_vercel_url

        # URL for the prediction code and status receiver
        self.start_prediction_url = f"{self.base_url}/api/predict"
        self.get_status_url = f"{self.base_url}/api/get-status"

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

    def predict(self, image_path, audio_path, guidance_scale=1.0, output_format="mp4", verbose=False):
        """
        Calls the prediction API asynchronously. First, it starts the prediction,
        then it polls for the result.
        """
        try:
            if verbose:
                print("Starting prediction...")
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

            start_response = requests.post(
                self.start_prediction_url,
                headers=headers,
                json=input_data
            )
            start_response.raise_for_status()
            
            prediction_data = start_response.json()
            prediction_id = prediction_data.get("id")

            if not prediction_id:
                raise ValueError("Failed to get prediction ID from the server.")
            
            if verbose:
                print(f"Prediction started successfully. ID: {prediction_id}")
                print("Polling for result...")

            start_time = time.time()
            while True:
                status_response = requests.get(
                    f"{self.get_status_url}?id={prediction_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                status_response.raise_for_status()
                result = status_response.json()

                status = result.get("status")
                if verbose:
                    elapsed_time = time.time() - start_time
                    print(f"Current status: {status} | Runtime: {elapsed_time:.2f}s", end='\r')

                if status == "succeeded":
                    if verbose:
                        print("\nPrediction succeeded!")
                    output_url = result.get("output")
                    
                    if output_format == "chunks":
                        return RawData(data_url=output_url)
                    else:
                        return output_url

                elif status == "failed":
                    if verbose:
                        print("\nPrediction failed.")
                    raise RuntimeError(f"Prediction failed with error: {result.get('error')}")
                
                time.sleep(5)

        except (requests.exceptions.RequestException, FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"\nAn error occurred: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Server responded with status {e.response.status_code}:")
                try:
                    print(json.dumps(e.response.json(), indent=2))
                except json.JSONDecodeError:
                    print(e.response.text)
            return None
