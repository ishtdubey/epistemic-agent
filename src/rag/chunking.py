from pathlib import Path


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Cut text into pieces of about chunk_size characters.
    Each piece overlaps the previous one by `overlap` characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be bigger than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be 0 or more, and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def chunk_file(path: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Read a .txt file and return chunks, each with its source file name."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    pieces = chunk_text(text, chunk_size, overlap)
    return [
        {"text": p, "source": path.name, "metadata": {"chunk_index": i}}
        for i, p in enumerate(pieces)
    ]