import pytest

from src.rag.chunking import chunk_file, chunk_text


def test_chunks_overlap():
    text = "abcdefghij" * 20
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_bad_settings_raise_error():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)


def test_chunk_file_adds_source_and_index(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world " * 30)
    result = chunk_file(str(f), chunk_size=100, overlap=10)
    assert result[0]["source"] == "note.txt"
    assert result[0]["metadata"]["chunk_index"] == 0
    assert result[1]["metadata"]["chunk_index"] == 1