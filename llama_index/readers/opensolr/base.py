"""Opensolr reader for LlamaIndex: local files read on Opensolr's servers, a PDF page by page."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from llama_index.vector_stores.opensolr.client import OpensolrClient, list_files

logger = logging.getLogger(__name__)


class OpensolrReader(BaseReader):
    """Load local files as Documents, read on Opensolr's servers.

    Each file is read there: its text, the text in its pictures (OCR, more than 100
    languages), a short description of the pictures that have no text, scanned pages
    and picture files (JPG, PNG, TIFF...) included. A PDF gives one Document per page
    (``metadata["page_label"]``), any other file one Document. Nothing is indexed.

    Example:
        .. code-block:: python

            from llama_index.readers.opensolr import OpensolrReader

            reader = OpensolrReader(index_name="mysite__dense", email="you@example.com",
                                    api_key="...", pattern="*.pdf")
            docs = reader.load_data("./contracts")

    Args:
        index_name: The Opensolr index the work is counted on: each picture read costs
            0.5 of an AI request; a file this account sent before is read for free.
        email: Opensolr account email.
        api_key: Opensolr API key.
        client: Optional pre-configured :class:`OpensolrClient` (overrides email/api_key).
        pattern: Which files of a folder, e.g. ``"*.pdf"``.
        recursive: Walk the sub-folders too.
        timeout: Seconds to wait for each batch of files to be read.

    Files that could not be read are skipped and listed in ``failed`` (path, error).
    Files still being read when ``timeout`` ran out are listed in ``pending``: load
    again to get them, their bytes are not sent twice.
    """

    #: Files sent, and waited for, together.
    BATCH = 20

    def __init__(
        self,
        index_name: str,
        email: str = "",
        api_key: str = "",
        client: Optional[OpensolrClient] = None,
        pattern: str = "*",
        recursive: bool = True,
        timeout: float = 600.0,
    ) -> None:
        if client is None:
            if not (email and api_key):
                raise ValueError("Provide either a client or email + api_key")
            client = OpensolrClient(email, api_key)
        if not index_name:
            raise ValueError("index_name is required")
        self._client = client
        self.index_name = index_name
        self.pattern = pattern
        self.recursive = recursive
        self.timeout = timeout
        self.failed: List[Dict[str, str]] = []
        self.pending: List[str] = []

    def lazy_load_data(self, path: Union[str, Path, None] = None, file_path: Union[str, Path, None] = None,
                       **load_kwargs: Any) -> Iterable[Document]:
        """``path`` (or ``file_path``): a file or a folder."""
        target = path if path is not None else file_path
        if target is None:
            raise ValueError("Give the file or folder to read: load_data(path)")
        self.failed, self.pending = [], []
        files = list_files(target, self.pattern, self.recursive)
        for i in range(0, len(files), self.BATCH):
            entries = self._client.extract_files(self.index_name, files[i : i + self.BATCH], pages=True, timeout=self.timeout)
            for entry in entries:
                yield from self._documents(entry)

    def _documents(self, entry: Dict[str, Any]) -> Iterable[Document]:
        state = entry.get("state")
        if state in ("queued", "processing"):
            self.pending.append(entry["path"])
            logger.warning("Opensolr is still reading %s; load again to get it", entry["path"])
            return
        if state != "done":
            error = str(entry.get("error") or state or "not read")
            self.failed.append({"path": entry["path"], "error": error})
            logger.warning("Opensolr could not read %s: %s", entry["path"], error)
            return
        base = {
            "file_path": entry["path"],
            "file_name": Path(entry["path"]).name,
            "file_type": entry.get("mime") or "",
        }
        pages = entry.get("pages") or []
        if pages:
            for p in pages:
                text = str(p.get("text") or "")
                if text.strip():
                    yield Document(text=text, metadata={
                        **base,
                        "page_label": str(int(p.get("page") or 0)),
                        "is_ocr": bool(p.get("is_ocr")),
                        "language": str(p.get("lang") or ""),
                    })
            return
        text = str(entry.get("text") or "")
        if text.strip():
            yield Document(text=text, metadata=base)
