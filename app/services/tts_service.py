import httpx
import logging
import time
from app.core.config import settings

logger = logging.getLogger(__name__)

CARTESIA_API_URL = "https://api.cartesia.ai/tts/bytes"


class TTSService:

    async def synthesize(self, text: str) -> tuple[bytes, float]:
        """
        Convert text to speech using Cartesia.
        Returns (audio_bytes, latency_ms)
        """
        start = time.time()

        headers = {
            "X-API-Key": settings.CARTESIA_API_KEY,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json"
        }

        payload = {
            "transcript": text,
            "model_id": settings.CARTESIA_MODEL,
            "voice": {
                "mode": "id",
                "id": settings.CARTESIA_VOICE_ID
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": 44100
            }
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(CARTESIA_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            audio_bytes = response.content

        latency_ms = (time.time() - start) * 1000
        logger.info(f"TTS latency: {latency_ms:.0f}ms | text length: {len(text)} chars")
        return audio_bytes, latency_ms


tts_service = TTSService()
