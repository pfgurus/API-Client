import requests
import json
import base64
import mimetypes
import os
import time

#---uncomment to use the utils inside handle_raw---
#from .handle_raw import RawData

class APIClient:
    def __init__(self, api_key, api_set="text_to_video", base_vercel_url="https://atv-model-api.vercel.app"):
        if not api_key:
            raise ValueError("An API key is required.")
        self.api_key = api_key
        self.base_url = base_vercel_url
        self.api_set = api_set

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

    def _print_verbose_metrics(self, result):
        """
        Parses and prints timing metrics from a successful prediction result.
        """
        metrics = result.get("metrics")
        if not metrics:
            return  #

        print("\n--- Prediction Metrics ---")

        metric_keys_map = {
            "tts_queue_time": "TTS Queue Time",
            "tts_inference_time": "TTS Inference Time",
            "animation_queue_time": "Animation Queue Time",
            "animation_inference_time": "Animation Inference Time"
        }

        found_detailed_metrics = False
        for key, label in metric_keys_map.items():
            if key in metrics:
                print(f"  {label + ':':<28} {metrics[key]:.2f}s")
                found_detailed_metrics = True
        
        # Fallback to standard Replicate metrics if no detailed keys are found
        if not found_detailed_metrics and "predict_time" in metrics:
            predict_time = metrics.get("predict_time", 0)
            total_time = metrics.get("total_time", 0)
            
            queue_and_setup_time = total_time - predict_time

            print(f"  {'Queue & Setup Time:':<28} {queue_and_setup_time:.2f}s")
            print(f"  {'Inference Time:':<28} {predict_time:.2f}s")
            print("  (Standard metrics shown.)")

        print("--------------------------\n")

    def predict(self, image_path=None, audio_path=None, text=None, model="default", verbose=False, guidance_scale=1.0, **kwargs):
        """
        Main dispatcher. Calls the correct method based on the api_set attribute.
        """
        metrics_out = kwargs.pop("metrics_out", None)
        
        api_set = kwargs.pop("api_set", None) or self.api_set

        if api_set == "text_to_video":
            if not text or not image_path:
                raise ValueError("Text-to-video requires 'text' and 'image_path'.")
            return self.generate_video_from_text(image_path, text, model, verbose=verbose, guidance_scale=guidance_scale, **kwargs)
        else:
            if not image_path or not audio_path:
                raise ValueError("Video generation requires 'image_path' and 'audio_path'.")
            if model == "atv_stream":
                return self._predict_stream(image_path, audio_path, model, verbose=verbose, guidance_scale=guidance_scale, metrics_out=metrics_out, **kwargs)
            else:
                return self._predict_batch(image_path, audio_path, model, verbose=verbose, guidance_scale=guidance_scale, **kwargs)


    def _predict_batch(self, image_path, audio_path, model, verbose, output_format="mp4", guidance_scale=1.0, **kwargs):
        """
        Predict logic for non-streaming models. Polls for a single result.
        Returns a tuple: (output, metrics)
        """
        try:
            if verbose: print(f"Starting prediction with model: {model} (batch mode)...")
            image_uri = self._file_to_data_uri(image_path)
            audio_uri = self._file_to_data_uri(audio_path)
            input_data = { "model": model, "image_input": image_uri, "audio_input": audio_uri, "guidance_scale": guidance_scale, "output_format": output_format }
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
                    metrics = result.get("metrics", {}) 
                    if verbose: 
                        print("\nPrediction succeeded!")
                        self._print_verbose_metrics(result)
                    
                    output_url = result.get("output")
                    if output_format == "chunks":
                        try:
                            from .handle_raw import RawData
                        except Exception as import_err:
                            raise ImportError(
                                "RawData is not available..." 
                            ) from import_err
                        return RawData(data_url=output_url), metrics # Return tuple
                    else:
                        return output_url, metrics # Return tuple
                
                elif status == "failed":
                    if verbose: print("\nPrediction failed.")
                    raise RuntimeError(f"Prediction failed with error: {result.get('error')}")
                
                time.sleep(5)
        except (requests.exceptions.RequestException, FileNotFoundError, ValueError, RuntimeError) as e:
            self._handle_exception(e)
            return None, {} # Return empty metrics on failure

    def generate_video_from_text(self, image_path, text, model="default", voice_id="Deep_Voice_Man", language_boost=None, emotion=None, verbose=False, guidance_scale=1.0, **kwargs):
        """
        Initiates and waits for the text-to-video process to complete.
        Returns a tuple: (output, metrics)
        """
        try:
            if verbose: print(f"Starting text-to-video generation (synchronous)...")
            image_uri = self._file_to_data_uri(image_path)
            
            input_data = {
                "model": model,
                "api_set": "text_to_video",
                "image_input": image_uri,
                "text_input": text,
                "voice_id": voice_id,
                "language_boost": language_boost,
                "emotion": emotion,
                "guidance_scale": guidance_scale,
            }
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

            start_response = requests.post(
                self.start_prediction_url, 
                headers=headers, 
                json=input_data,
                timeout=600 # 10 minute timeout
            )
            start_response.raise_for_status() # This will raise an error if status is 4xx or 5xx
            
            result = start_response.json()
            status = result.get("status")

            if status == "succeeded":
                metrics = result.get("metrics", {})
                if verbose: 
                    print("\nProcess finished successfully.")
                    self._print_verbose_metrics(result) 
                
                # Return the final output URL and the combined metrics
                return result.get("output"), metrics
            
            elif status == "failed":
                if verbose: print("\nProcess failed.")
                raise RuntimeError(f"Prediction failed with error: {result.get('error')}")
            else:
                raise RuntimeError(f"Unexpected status received from synchronous call: {status}")

        except requests.exceptions.Timeout:
            print("\nError: The request timed out (10 minutes).")
            print("Your Vercel function may have hit its execution limit, or the model took too long.")
            raise
        except Exception as e:
            self._handle_exception(e)
            raise

    def _predict_stream(self, image_path, audio_path, model, verbose, guidance_scale=1.0, output_format="mp4", **kwargs):
        """
        Yields chunk URLs as they become available.
        Populates 'metrics_out' dict if provided.
        """
        metrics_out = kwargs.pop("metrics_out", None)

        try:
            if verbose: print(f"Starting prediction with model: {model} (stream mode)...")
            image_uri = self._file_to_data_uri(image_path)
            audio_uri = self._file_to_data_uri(audio_path)
            input_data = { "model": model, "image_input": image_uri, "audio_input": audio_uri, "guidance_scale": guidance_scale, "output_format": output_format }
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
                    if verbose: 
                        print("\nPrediction stream finished successfully.")
                        self._print_verbose_metrics(result)
                    
                    if metrics_out is not None and isinstance(metrics_out, dict):
                        metrics_out.update(result.get("metrics", {}))
                    break 
                
                elif status == "failed":
                    if verbose: print("\nPrediction failed.")
                    raise RuntimeError(f"Prediction failed with error: {result.get('error')}")
                
                time.sleep(2)
        except (requests.exceptions.RequestException, FileNotFoundError, ValueError, RuntimeError) as e:
            self._handle_exception(e)
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
