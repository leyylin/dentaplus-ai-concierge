import asyncio
import subprocess

import imageio_ffmpeg
import numpy as np
from faster_whisper import WhisperModel
from loguru import logger


def _decode_audio_ffmpeg(audio_bytes: bytes) -> np.ndarray:
    """Конвертирует аудио байты в 16kHz моно float32 numpy массив через FFmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-threads", "0",
        "-i", "pipe:0",
        "-f", "s16le",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1"
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = process.communicate(input=audio_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {err.decode('utf-8', errors='ignore')}")

    # Нормализация int16 в float32 (-1.0 .. 1.0), как ожидает Whisper
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio


class SpeechToText:
    """Класс для локального распознавания речи через faster-whisper."""

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8") -> None:
        logger.info(
            "Initializing WhisperModel (model_size={}, device={}, compute_type={})",
            model_size,
            device,
            compute_type,
        )
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logger.info("WhisperModel initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize WhisperModel: {}", exc)
            raise

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        """Синхронный метод транскрибации аудио."""
        try:
            audio_np = _decode_audio_ffmpeg(audio_bytes)
            initial_prompt = "Стоматологическая клиника Дента-Плюс, запись на приём, профессиональная гигиена, отбеливание зубов, кариес, консультация, терапевт, хирург."
            segments, _ = self.model.transcribe(
                audio_np,
                language="ru",
                beam_size=5,
                initial_prompt=initial_prompt
            )
            return " ".join([segment.text for segment in segments]).strip()
        except Exception as e:
            logger.exception("STT processing error: {}", e)
            return ""

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Асинхронный метод транскрибации аудио без блокировки event loop."""
        try:
            return await asyncio.to_thread(self._transcribe_sync, audio_bytes)
        except Exception as e:
            logger.exception("STT processing error: {}", e)
            return ""
