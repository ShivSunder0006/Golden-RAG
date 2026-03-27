"""
transcript_extractor.py — YouTube Transcript Extraction Module

Fetches, cleans, and structures transcripts from 4 3Blue1Brown neural network videos.
Uses youtube-transcript-api v1.2+ (instance-based API) with JSON caching.
"""

import json
import os
from datetime import timedelta
from youtube_transcript_api import YouTubeTranscriptApi

# ─── Video Metadata ────────────────────────────────────────────────────────────

VIDEOS = {
    "aircAruvnKk": {
        "title": "But what is a neural network? | Chapter 1, Deep Learning",
        "url": "https://www.youtube.com/watch?v=aircAruvnKk",
    },
    "wjZofJX0v4M": {
        "title": "Gradient descent, how neural networks learn | Chapter 2, Deep Learning",
        "url": "https://www.youtube.com/watch?v=wjZofJX0v4M",
    },
    "fHF22Wxuyw4": {
        "title": "What is backpropagation really doing? | Chapter 3, Deep Learning",
        "url": "https://www.youtube.com/watch?v=fHF22Wxuyw4",
    },
    "C6YtPJxNULA": {
        "title": "Neural Networks from Scratch in Hindi/Urdu",
        "url": "https://www.youtube.com/watch?v=C6YtPJxNULA",
    },
}

# ─── Cache Directory ───────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transcripts")


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to a human-readable HH:MM:SS or MM:SS timestamp."""
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _fetch_single_transcript(video_id: str) -> list[dict]:
    """
    Fetch transcript for a single video using youtube-transcript-api v1.2+.
    
    Uses instance-based API: YouTubeTranscriptApi().fetch(video_id)
    Tries English first, then Hindi, then any available language.
    """
    ytt = YouTubeTranscriptApi()

    # Strategy 1: Try fetching with language preferences
    for languages in [["en"], ["hi"], ["en", "hi"]]:
        try:
            transcript = ytt.fetch(video_id, languages=languages)
            return [
                {
                    "text": snippet.text,
                    "start": snippet.start,
                    "duration": snippet.duration,
                    "timestamp": _format_timestamp(snippet.start),
                }
                for snippet in transcript
            ]
        except Exception:
            continue

    # Strategy 2: List available transcripts and pick the first one
    try:
        transcript_list = ytt.list(video_id)
        for t in transcript_list:
            try:
                fetched = t.fetch()
                return [
                    {
                        "text": snippet.text,
                        "start": snippet.start,
                        "duration": snippet.duration,
                        "timestamp": _format_timestamp(snippet.start),
                    }
                    for snippet in fetched
                ]
            except Exception:
                continue
    except Exception as e:
        print(f"  ⚠ Could not list transcripts for {video_id}: {e}")

    # Strategy 3: Try direct fetch without language filter
    try:
        transcript = ytt.fetch(video_id)
        return [
            {
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration,
                "timestamp": _format_timestamp(snippet.start),
            }
            for snippet in transcript
        ]
    except Exception as e:
        print(f"  ⚠ Final fallback failed for {video_id}: {e}")

    return []


def _merge_transcript_text(segments: list[dict]) -> str:
    """Merge transcript segments into a single cleaned text string."""
    texts = []
    for seg in segments:
        text = seg["text"].strip()
        # Clean up common artifacts from auto-generated captions
        text = text.replace("\n", " ").replace("  ", " ")
        if text:
            texts.append(text)
    return " ".join(texts)


def fetch_transcripts(force_refresh: bool = False) -> dict:
    """
    Fetch transcripts for all configured videos.

    Returns:
        dict: {video_id: {title, url, segments, full_text, segment_count}}
    
    Transcripts are cached to disk as JSON for fast reload.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    transcripts = {}

    for video_id, meta in VIDEOS.items():
        cache_path = os.path.join(DATA_DIR, f"{video_id}.json")

        # Try loading from cache
        if not force_refresh and os.path.exists(cache_path):
            print(f"  ✓ Loading cached transcript: {meta['title'][:50]}...")
            with open(cache_path, "r", encoding="utf-8") as f:
                transcripts[video_id] = json.load(f)
            continue

        # Fetch fresh transcript from YouTube
        print(f"  ⏳ Fetching transcript: {meta['title'][:50]}...")
        segments = _fetch_single_transcript(video_id)

        if not segments:
            print(f"  ⚠ No transcript available for: {meta['title']}")
            continue

        transcript_data = {
            "title": meta["title"],
            "url": meta["url"],
            "video_id": video_id,
            "segments": segments,
            "full_text": _merge_transcript_text(segments),
            "segment_count": len(segments),
        }

        # Cache to disk
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

        transcripts[video_id] = transcript_data
        print(f"  ✓ Fetched {len(segments)} segments")

    print(f"\n📚 Total transcripts loaded: {len(transcripts)}")
    return transcripts


def get_transcript_for_display(transcripts: dict, video_id: str) -> str:
    """Format a transcript for display with timestamps."""
    if video_id not in transcripts:
        return "Transcript not available."

    data = transcripts[video_id]
    lines = []
    for seg in data["segments"]:
        lines.append(f"[{seg['timestamp']}]  {seg['text']}")
    return "\n".join(lines)


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🎬 Fetching YouTube Transcripts...\n")
    t = fetch_transcripts()
    for vid, data in t.items():
        print(f"  {data['title']}: {data['segment_count']} segments, {len(data['full_text'])} chars")
