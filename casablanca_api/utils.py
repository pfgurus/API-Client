from torchaudio.io import StreamWriter
import torch


def save_av_clip(frames, audio, output_path, video_frame_rate=25, audio_sample_rate=16000):
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