import requests
import json
import base64
import mimetypes
import os
import time
from handle_raw import RawData

class APIClient:
    def __init__(self, api_key, base_vercel_url="https://atv-model-api.vercel.app"):
        if not api_key:
            raise ValueError("An API key is required.")
        self.api_key = api_key
        self.base_url = base_vercel_url

        self.start_prediction_url = f"{self.base_url}/api/predict"
        self.get_status_url = f"{self.base_url}/api/get-status"
        self.list_models_url = f"{self.base_url}/api/list-models"

    def list_models(self, display=False):
        """
        Fetches the curated list of available models from the API.
        This call does not require authentication.
        """
        try:
            response = requests.get(self.list_models_url)
            response.raise_for_status()
            models = response.json()

            if display and models:
                print("--- Available Models ---")
                for model in models:
                    print(f"\nID: {model['id']}")
                    print(f"Name: {model['name']}")
                    if 'description' in model:
                        print(f"Description: {model['description']}")
                    if 'price_per_second' in model:
                        price_per_hour = model['price_per_second'] * 3600
                        print(f"Price: ${price_per_hour:.2f}/hour (${model['price_per_second']}/second)")
                print("\n----------------------")
            
            return models
        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred while fetching models: {e}")
            return None

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

    def predict(self, image_path, audio_path, model="default", guidance_scale=1.0, output_format="mp4", verbose=False):
        """
        Calls the appropriate prediction method based on the model name.
        - For "atv_stream", it streams the results.
        - For all other models, it polls and returns a single final result.
        """
        if model == "atv_stream":
            # Note: We are ignoring 'output_format' for the stream model
            return self._predict_stream(image_path, audio_path, model, guidance_scale, verbose)
        else:
            # Call the regular method for batch predictions
            return self._predict_batch(image_path, audio_path, model, guidance_scale, output_format, verbose)

    def _predict_batch(self, image_path, audio_path, model, guidance_scale, output_format, verbose):
        """
        Predict logic for non-streaming models. Polls for a single result.
        """
        try:
            if verbose: print(f"Starting prediction with model: {model} (batch mode)...")
            image_uri = self._file_to_data_uri(image_path)
            audio_uri = self._file_to_data_uri(audio_path)

            input_data = {
                "model": model, "source_image": image_uri, "audio": audio_uri,
                "guidance_scale": guidance_scale, "output_format": output_format
            }
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

            start_response = requests.post(self.start_prediction_url, headers=headers, json=input_data)
            start_response.raise_for_status()
            
            prediction_id = start_response.json().get("id")
            if not prediction_id: raise ValueError("Failed to get prediction ID from the server.")
            
            if verbose: print(f"Prediction started. ID: {prediction_id}. Polling for result...")

            start_time = time.time()
            while True:
                status_response = requests.get(f"{self.get_status_url}?id={prediction_id}", headers=headers)
                status_response.raise_for_status()
                result = status_response.json()
                status = result.get("status")

                if verbose:
                    elapsed_time = time.time() - start_time
                    print(f"Current status: {status} | Runtime: {elapsed_time:.2f}s", end='\r')

                if status == "succeeded":
                    if verbose: print("\nPrediction succeeded!")
                    output_url = result.get("output")
                    return RawData(data_url=output_url) if output_format == "chunks" else output_url
                elif status == "failed":
                    if verbose: print("\nPrediction failed.")
                    raise RuntimeError(f"Prediction failed with error: {result.get('error')}")
                
                time.sleep(5)
        except (requests.exceptions.RequestException, FileNotFoundError, ValueError, RuntimeError) as e:
            self._handle_exception(e)
            return None
            
    def _predict_stream(self, image_path, audio_path, model, guidance_scale, verbose):
        """
        Yields chunk URLs as they become available.
        """
        try:
            if verbose: print(f"Starting prediction with model: {model} (stream mode)...")
            image_uri = self._file_to_data_uri(image_path)
            audio_uri = self._file_to_data_uri(audio_path)

            input_data = {"model": model, "source_image": image_uri, "audio": audio_uri, "guidance_scale": guidance_scale}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

            start_response = requests.post(self.start_prediction_url, headers=headers, json=input_data)
            start_response.raise_for_status()
            
            prediction_id = start_response.json().get("id")
            if not prediction_id: raise ValueError("Failed to get prediction ID from the server.")

            if verbose: print(f"Prediction started. ID: {prediction_id}. Polling for stream...")

            yielded_chunk_count = 0
            start_time = time.time()
            
            while True:
                status_response = requests.get(f"{self.get_status_url}?id={prediction_id}", headers=headers)
                status_response.raise_for_status()
                result = status_response.json()
                status = result.get("status")
                
                if verbose:
                    elapsed_time = time.time() - start_time
                    print(f"Status: {status} | Chunks received: {yielded_chunk_count} | Runtime: {elapsed_time:.2f}s", end='\r')
                
                output_chunks = result.get("output", [])
                
                if isinstance(output_chunks, list) and len(output_chunks) > yielded_chunk_count:
                    new_chunks = output_chunks[yielded_chunk_count:]
                    for chunk_url in new_chunks:
                        yield chunk_url
                    yielded_chunk_count = len(output_chunks)
                
                if status == "succeeded":
                    if verbose: print("\nPrediction stream finished successfully.")
                    break
                elif status == "failed":
                    if verbose: print("\nPrediction failed.")
                    raise RuntimeError(f"Prediction failed with error: {result.get('error')}")
                
                time.sleep(2)
        except (requests.exceptions.RequestException, FileNotFoundError, ValueError, RuntimeError) as e:
            self._handle_exception(e)
            # A generator should handle exceptions, so we re-raise it for the caller.
            raise

    def _handle_exception(self, e):
        """Centralized exception handling for predict methods."""
        print(f"\nAn error occurred: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Server responded with status {e.response.status_code}:")
            try:
                print(json.dumps(e.response.json(), indent=2))
            except json.JSONDecodeError:
                print(e.response.text)
