import os
import time
from typing import Iterable, List, Tuple, Dict, Any

from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_core.documents import Document
import serpapi


load_dotenv()


def _get_youtube_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is not set in the environment")

    return build("youtube", "v3", developerKey=api_key)


def _get_serpapi_client():
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise ValueError("SERPAPI_API_KEY is not set in the environment")

    return serpapi.Client(api_key=api_key)


def search_youtube_videos(
    query: str,
    max_results: int = 3,
    published_after: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Search YouTube and return a list of simple video metadata dicts.
    """
    youtube = _get_youtube_client()

    params: Dict[str, Any] = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
    }

    if published_after:
        params["publishedAfter"] = published_after

    response = youtube.search().list(**params).execute()

    videos_metadata: List[Dict[str, Any]] = []
    for item in response.get("items", []):
        videos_metadata.append(
            {
                "videoId": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "publishedAt": item["snippet"]["publishedAt"],
            }
        )

    return videos_metadata


def fetch_video_transcripts(
    videos_metadata: Iterable[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Any]]:
    """
    For each video in `videos_metadata`, fetch an English transcript using
    SerpAPI's `youtube_video_transcript` engine.

    Returns a list of (video_metadata, transcript_segments) tuples where
    `transcript_segments` is SerpAPI's `transcript` array.
    """

    client = _get_serpapi_client()

    def _fetch_one(video: Dict[str, Any]):
        try:
            time.sleep(1.0)  # SerpAPI is rate-limited; keep requests spaced out

            payload: Dict[str, Any] = {
                "engine": "youtube_video_transcript",
                "v": video["videoId"],
                "type": "asr",
                "language_code": "en",
            }
            data = client.search(payload)
            transcript = data.get("transcript") or []

            if not transcript:
                print(f"⚠ No transcript segments for: {video['title']}")
                return None

            print(f"✅ Got transcript: {video['title']}")
            # transcript is an array of objects with keys like:
            # start_ms, end_ms, snippet, start_time_text
            return (video, transcript)

        except Exception as e:
            print(f"❌ Error ({video['title']}): {e}")
            return None

    results = list(map(_fetch_one, videos_metadata))
    videos_with_transcripts = [r for r in results if r is not None]

    print(f"\n🎉 Total videos with transcripts: {len(videos_with_transcripts)}")
    return videos_with_transcripts


def _format_timestamps(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_docs_from_transcripts(
    videos_with_transcripts: Iterable[Tuple[Dict[str, Any], Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[Document]:
    """
    Convert SerpAPI transcript segments into LangChain `Document` chunks.

    Each transcript segment uses SerpAPI keys:
      - `snippet`: transcript text
      - `start_ms`: segment start time in milliseconds
    """
    docs: List[Document] = []

    for video, fetched in videos_with_transcripts:
        segments = fetched
        if not segments:
            continue

        chunk_text = ""
        chunk_start_ms = segments[0].get("start_ms")

        def _coerce_ms(value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return int(value)
            try:
                return int(float(str(value)))
            except Exception:
                return None

        for seg in segments:
            snippet = seg.get("snippet")
            if not snippet:
                continue

            seg_start_ms = _coerce_ms(seg.get("start_ms"))
            chunk_text += " " + snippet

            if len(chunk_text) >= chunk_size:
                ts_ms = _coerce_ms(chunk_start_ms) or seg_start_ms or 0
                timestamp_secs = int(ts_ms / 1000)

                docs.append(
                    Document(
                        page_content=chunk_text.strip(),
                        metadata={
                            "title": video["title"],
                            "channel": video["channel"],
                            "publishedAt": video["publishedAt"],
                            "timestamp": _format_timestamps(timestamp_secs),
                            "url_with_timestamp": (
                                f"{video['link']}&t={timestamp_secs}s"
                            ),
                        },
                    )
                )
                chunk_text = chunk_text[-overlap:]
                chunk_start_ms = seg_start_ms or chunk_start_ms

        if chunk_text.strip():
            ts_ms = _coerce_ms(chunk_start_ms) or 0
            timestamp_secs = int(ts_ms / 1000)
            docs.append(
                Document(
                    page_content=chunk_text.strip(),
                    metadata={
                        "title": video["title"],
                        "channel": video["channel"],
                        "publishedAt": video["publishedAt"],
                        "timestamp": _format_timestamps(timestamp_secs),
                        "url_with_timestamp": (
                            f"{video['link']}&t={timestamp_secs}s"
                        ),
                    },
                )
            )

    return docs

