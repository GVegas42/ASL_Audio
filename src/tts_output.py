"""
Background text-to-speech worker so recognized signs can be spoken aloud
without blocking the video/inference loop.
"""

import queue
import threading
from typing import Optional


class TTSEngine:
    """Queues text and speaks it asynchronously on a background thread."""

    def __init__(self, rate: int = 150, volume: float = 1.0):
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._rate = rate
        self._volume = volume
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str):
        """Queue `text` to be spoken. Non-blocking."""
        if text:
            self._queue.put(text)

    def stop(self):
        """Stop the worker thread, discarding anything left in the queue."""
        self._queue.put(None)
        self._thread.join(timeout=2)

    def _run(self):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty('rate', self._rate)
        engine.setProperty('volume', self._volume)

        while True:
            text = self._queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
