from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Chunk:
    """统一块结构。"""

    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, object] = field(default_factory=dict)


def _tokens(text: str) -> List[str]:
    return text.split()


def _count_tokens(text: str) -> int:
    return len(_tokens(text))


def _detect_type(text: str) -> str:
    """根据文本形态识别 text/table/image。"""
    t = text.strip().lower()
    if t.startswith("|") or ("| " in t and t.count("|") >= 2):
        return "table"
    if t.startswith("table:") or t.startswith("| "):
        return "table"
    if t.startswith("image:") or t.startswith("figure:") or t.startswith("!['"):
        return "image"
    return "text"


def _heading_level(line: str) -> Optional[int]:
    """识别 Markdown H1/H2 标题层级。"""
    stripped = line.strip()
    if stripped.startswith("## "):
        return 2
    if stripped.startswith("# "):
        return 1
    return None


@dataclass
class Section:
    """结构解析后的章节单元。"""

    title: str
    level: int
    content_lines: List[str] = field(default_factory=list)


def _parse_sections(text: str) -> List[Section]:
    """按 H1/H2 边界做结构粗解析；无标题时退化为单章节。"""
    lines = text.split("\n")
    sections: List[Section] = []
    current: Optional[Section] = None

    for line in lines:
        lvl = _heading_level(line)
        if lvl in (1, 2):
            if current is not None:
                sections.append(current)
            current = Section(title=line.strip().lstrip("# ").strip(), level=lvl)
        else:
            if current is None:
                current = Section(title="", level=0)
            current.content_lines.append(line)

    if current is not None:
        sections.append(current)

    return sections if sections else [Section(title="", level=0, content_lines=lines)]


def _coarse_chunks_from_section(
    doc_id: str,
    section: Section,
    structural_chunk_size: int,
    base_metadata: Dict[str, object],
    start_idx: int,
) -> List[Chunk]:
    """章节级粗分块（上限 structural_chunk_size），并保护图表原子块。"""
    chunks: List[Chunk] = []
    idx = start_idx
    current_lines: List[str] = []
    current_tokens = 0

    def flush(lines: List[str], atomic_type: str = "text") -> None:
        nonlocal idx
        text = " ".join(" ".join(lines).split())
        if not text.strip():
            return
        meta = dict(base_metadata)
        meta.update({
            "doc_id": doc_id,
            "section_title": section.title,
            "chunk_index": idx,
            "type": atomic_type,
            "semantic_chunk": False,
            "page_number": base_metadata.get("page_number", None),
        })
        chunks.append(Chunk(chunk_id=f"{doc_id}_coarse_{idx}", doc_id=doc_id, text=text, metadata=meta))
        idx += 1

    for line in section.content_lines:
        line = line.strip()
        if not line:
            continue
        ltype = _detect_type(line)
        if ltype in {"table", "image"}:
            if current_lines:
                flush(current_lines)
                current_lines = []
                current_tokens = 0
            flush([line], ltype)
        else:
            toks = _count_tokens(line)
            if current_tokens + toks > structural_chunk_size and current_lines:
                flush(current_lines)
                current_lines = [line]
                current_tokens = toks
            else:
                current_lines.append(line)
                current_tokens += toks

    if current_lines:
        flush(current_lines)

    return chunks


def _semantic_refine(
    coarse: List[Chunk],
    semantic_min_size: int,
    semantic_max_size: int,
) -> List[Chunk]:
    """对超长文本块做语义细分，保证块大小落入目标区间。"""
    refined: List[Chunk] = []
    for chunk in coarse:
        if chunk.metadata.get("type") in {"table", "image"}:
            refined.append(chunk)
            continue
        words = _tokens(chunk.text)
        if len(words) <= semantic_max_size:
            refined.append(chunk)
            continue

        i = 0
        sub_idx = 0
        while i < len(words):
            j = min(i + semantic_max_size, len(words))
            sub_text = " ".join(words[i:j])
            meta = dict(chunk.metadata)
            meta["semantic_chunk"] = True
            meta["chunk_index"] = int(str(chunk.metadata.get("chunk_index", 0))) * 1000 + sub_idx
            sub_id = f"{chunk.chunk_id}_sub{sub_idx}"
            refined.append(Chunk(chunk_id=sub_id, doc_id=chunk.doc_id, text=sub_text, metadata=meta))
            if j == len(words):
                break
            i = max(i + 1, j - semantic_min_size // 4)
            sub_idx += 1

    return refined


def _inject_overlap(
    chunks: List[Chunk],
    overlap_type: str,
    overlap_size: int,
) -> List[Chunk]:
    """在相邻文本块之间注入重叠信息，降低语义断裂。"""
    if len(chunks) <= 1 or overlap_size <= 0:
        return chunks

    result: List[Chunk] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        curr = chunks[i]
        if prev.metadata.get("type") in {"table", "image"} or curr.metadata.get("type") in {"table", "image"}:
            result.append(curr)
            continue

        if overlap_type == "sentence":
            import re

            sents = re.split(r"(?<=[.!?])\s+", prev.text)
            tail = " ".join(sents[-min(overlap_size, len(sents)):])
        else:
            toks = _tokens(prev.text)
            tail = " ".join(toks[-min(overlap_size, len(toks)):])

        new_text = (tail + " " + curr.text).strip()
        new_meta = dict(curr.metadata)
        new_meta["overlap_from"] = prev.chunk_id
        result.append(Chunk(chunk_id=curr.chunk_id, doc_id=curr.doc_id, text=new_text, metadata=new_meta))

    return result


def chunk_document(
    doc_id: str,
    text: str,
    strategy: str = "sentence",
    size: int = 512,
    overlap_type: str = "token",
    overlap_size: int = 100,
    semantic_min_size: int = 350,
    semantic_max_size: int = 650,
    preserve_table_as_markdown: bool = True,
    generate_image_caption: bool = False,
    window_size: int = 3,
    similarity_threshold: float = 0.65,
    base_metadata: Optional[Dict[str, object]] = None,
) -> List[Chunk]:
    """完整分块流程：结构解析→粗分块→语义细分→重叠注入。"""
    base_metadata = dict(base_metadata or {})
    base_metadata.setdefault("section_title", base_metadata.pop("title", ""))

    sections = _parse_sections(text)

    coarse: List[Chunk] = []
    idx = 0
    structural_chunk_size = 1000
    for section in sections:
        new_chunks = _coarse_chunks_from_section(
            doc_id=doc_id,
            section=section,
            structural_chunk_size=structural_chunk_size,
            base_metadata=base_metadata,
            start_idx=idx,
        )
        idx += len(new_chunks)
        coarse.extend(new_chunks)

    refined = _semantic_refine(coarse, semantic_min_size, semantic_max_size) if strategy == "semantic" else coarse
    final = _inject_overlap(refined, overlap_type, overlap_size)

    if not final:
        meta = dict(base_metadata)
        meta.update({"doc_id": doc_id, "chunk_index": 0, "type": "text", "semantic_chunk": False})
        return [Chunk(chunk_id=f"{doc_id}_chunk_0", doc_id=doc_id, text=text, metadata=meta)]

    return final
