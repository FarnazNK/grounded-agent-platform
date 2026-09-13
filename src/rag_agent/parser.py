from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

from rag_agent.config import Settings, get_settings
from rag_agent.errors import InvalidUploadError


@dataclass
class ParsedDocument:
    source_name: str
    media_type: str
    file_size: int
    sha256: str
    text: str
    metadata: dict[str, object]


@dataclass
class ParsedChunk:
    chunk_index: int
    content: str
    token_count: int
    metadata: dict[str, object]


PDF_MAGIC = b'%PDF'


def detect_media_type(filename: str, claimed_type: str | None, data: bytes, settings: Settings) -> str:
    inferred = mimetypes.guess_type(filename)[0]
    media_type = claimed_type or inferred or 'application/octet-stream'
    if data.startswith(PDF_MAGIC):
        media_type = 'application/pdf'
    if media_type not in settings.allowed_mime_types:
        raise InvalidUploadError(
            'Unsupported media type.',
            details={'media_type': media_type, 'allowed_types': list(settings.allowed_mime_types)},
        )
    return media_type


def parse_document_bytes(
    filename: str,
    claimed_type: str | None,
    data: bytes,
    *,
    settings: Settings | None = None,
) -> ParsedDocument:
    settings = settings or get_settings()
    if not data:
        raise InvalidUploadError('Uploaded file is empty.')
    if len(data) > settings.upload_max_bytes:
        raise InvalidUploadError(
            'Uploaded file exceeds size limit.',
            details={'max_bytes': settings.upload_max_bytes, 'actual_bytes': len(data)},
        )
    media_type = detect_media_type(filename, claimed_type, data, settings)
    if media_type == 'application/pdf':
        reader = PdfReader(BytesIO(data))
        text = '\n'.join((page.extract_text() or '').strip() for page in reader.pages).strip()
    elif media_type == 'application/json':
        parsed = json.loads(data.decode('utf-8'))
        text = json.dumps(parsed, indent=2, sort_keys=True)
    else:
        try:
            text = data.decode('utf-8').strip()
        except UnicodeDecodeError as exc:
            raise InvalidUploadError('Only UTF-8 text uploads are supported for text files.') from exc
    if not text:
        raise InvalidUploadError('Uploaded document did not contain extractable text.')
    return ParsedDocument(
        source_name=filename,
        media_type=media_type,
        file_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        text=text,
        metadata={'byte_size': len(data)},
    )


def chunk_text(text: str, *, settings: Settings | None = None) -> list[ParsedChunk]:
    settings = settings or get_settings()
    words = text.split()
    if not words:
        return []
    size = settings.chunk_size_words
    overlap = min(settings.chunk_overlap_words, max(0, size - 1))
    chunks: list[ParsedChunk] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(len(words), start + size)
        piece = ' '.join(words[start:end]).strip()
        if piece:
            chunks.append(
                ParsedChunk(
                    chunk_index=index,
                    content=piece,
                    token_count=len(piece.split()),
                    metadata={'word_start': start, 'word_end': end},
                )
            )
            index += 1
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks
