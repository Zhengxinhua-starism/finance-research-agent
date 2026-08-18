"""文本切分。

解决什么问题
    知识库里的内容是"行业分析框架""风险预警规则""披露制度"这类结构化短文。
    如果按固定字符数硬切，会把"应收账款增速远超营收增速 → 可能虚增收入"
    这条规则从箭头处切成两半，两个 chunk 各自都失去意义，检索出来也没法用。
    本模块保证切分点落在语义边界上。

核心设计决策
    1. 分隔符按语义强度分级递归切分：段落（\\n\\n）→ 换行 → 句号 → 分号 →
       逗号 → 字符。只有上一级切完还超长才降级到下一级。这样绝大多数 chunk
       的边界落在段落或句子上，而不是词的中间。
    2. 按**字符数**而不是 token 数计算长度。中文场景下 1 字 ≈ 1 token，
       字符计数与 token 计数几乎等价，但省掉了每次切分都跑一遍分词的开销
       （建库时要切几百次）。
    3. 保留重叠（默认 80 字符）。重叠的作用是让跨 chunk 的指代（"该指标"
       "上述规则"）在至少一个 chunk 里能找到指代对象。重叠取在句子边界上，
       不是硬切 80 个字。
    4. Markdown 标题会被下沉进每个 chunk 的元数据和正文前缀。
       "## 银行业分析框架"这个标题只出现在第一个 chunk 里的话，
       后续 chunk 检索出来时读者（和 LLM）不知道它在讲哪个行业。

为什么不用其他方案
    - 不用 LangChain 的 RecursiveCharacterTextSplitter：功能上等价，
      但引入它就要引入 langchain-text-splitters 包；而且它的中文分隔符
      默认列表不含"；""、"，效果反而要自己调。60 行自己写更可控。
    - 不用语义切分（按 embedding 相似度找断点）：知识库文档本身很短且结构清晰，
      语义切分的收益接近于零，成本却是每次建库都要跑一遍 embedding。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from config import get_config
from harness.types import Document

logger = logging.getLogger(__name__)

# 按语义强度从强到弱排列。切分时优先用强分隔符。
SEPARATORS: tuple[str, ...] = (
    "\n\n",  # 段落
    "\n",  # 换行
    "。",  # 句号
    "！",
    "？",
    "；",  # 分号
    ". ",  # 英文句号（带空格，避免切开小数）
    "，",  # 逗号（弱，只在前面都不行时用）
    "、",
    " ",
    "",  # 兜底：硬切
)

MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", flags=re.MULTILINE)


class TextChunker:
    """递归字符切分器。"""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: Sequence[str] = SEPARATORS,
    ):
        config = get_config()
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 chunk_size({self.chunk_size})，"
                "否则切分会陷入死循环"
            )
        self.separators = tuple(separators)

    # ---------------- 公开接口 ----------------

    def split_text(self, text: str) -> list[str]:
        """把长文本切成 chunk 列表。"""
        cleaned = self._normalize(text)
        if not cleaned:
            return []
        if len(cleaned) <= self.chunk_size:
            return [cleaned]
        pieces = self._recursive_split(cleaned, self.separators)
        return self._merge_with_overlap(pieces)

    def split_document(self, document: Document) -> list[Document]:
        """切分单个 Document，元数据继承并补上 chunk 序号。"""
        chunks = self.split_text(document.content)
        if len(chunks) <= 1:
            return [document]

        heading = self._extract_leading_heading(document.content)
        results: list[Document] = []
        for index, chunk in enumerate(chunks):
            # 给非首个 chunk 补标题前缀，保证脱离上下文后仍知道在讲什么
            content = chunk
            if heading and index > 0 and heading not in chunk:
                content = f"{heading}（续 {index + 1}/{len(chunks)}）\n{chunk}"
            metadata: dict[str, Any] = {
                **document.metadata,
                "parent_doc_id": document.doc_id,
                "chunk_index": index,
                "chunk_total": len(chunks),
            }
            if heading:
                metadata["heading"] = heading
            results.append(
                Document(
                    doc_id=f"{document.doc_id}#c{index}",
                    content=content,
                    metadata=metadata,
                )
            )
        return results

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        logger.info("切分完成: %d 篇文档 → %d 个 chunk", len(documents), len(chunks))
        return chunks

    def split_markdown(self, text: str, base_metadata: dict[str, Any] | None = None) -> list[Document]:
        """按 Markdown 标题先分节，再对超长节做递归切分。

        建库脚本从 md 文件导入知识时用这个入口：标题天然就是最好的语义边界，
        先按标题分能大幅提升 chunk 的内聚性。
        """
        base_metadata = base_metadata or {}
        sections = self._split_by_heading(text)
        documents: list[Document] = []
        for heading, body in sections:
            content = f"{heading}\n{body}".strip() if heading else body.strip()
            if not content:
                continue
            documents.append(
                Document(
                    content=content,
                    metadata={**base_metadata, "heading": heading or ""},
                )
            )
        return self.split_documents(documents)

    # ---------------- 内部实现 ----------------

    @staticmethod
    def _normalize(text: str) -> str:
        """统一空白：Windows 换行、全角空格、连续空行。"""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _recursive_split(self, text: str, separators: Sequence[str]) -> list[str]:
        """用当前最强的分隔符切，仍超长的片段递归用下一级分隔符切。"""
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            # 分隔符用尽，硬切
            return [
                text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
            ]

        separator, rest = separators[0], separators[1:]
        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)
        # 把分隔符补回去（除了最后一段），否则句号会被吃掉
        parts = [part + separator for part in parts[:-1]] + [parts[-1]]

        pieces: list[str] = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) <= self.chunk_size:
                pieces.append(part)
            else:
                pieces.extend(self._recursive_split(part, rest))
        return pieces

    def _merge_with_overlap(self, pieces: Sequence[str]) -> list[str]:
        """把小片段拼成接近 chunk_size 的 chunk，相邻 chunk 之间保留重叠。"""
        chunks: list[str] = []
        buffer: list[str] = []
        buffer_length = 0

        for piece in pieces:
            piece_length = len(piece)
            if buffer_length + piece_length > self.chunk_size and buffer:
                chunk = "".join(buffer).strip()
                if chunk:
                    chunks.append(chunk)
                overlap_parts = self._take_overlap(buffer)
                buffer = list(overlap_parts)
                buffer_length = sum(len(part) for part in buffer)
            buffer.append(piece)
            buffer_length += piece_length

        tail = "".join(buffer).strip()
        if tail:
            # 尾部太短就并进上一个 chunk，避免产生只有几个字的碎片 chunk。
            # 碎片 chunk 在 BM25 里会因为文档极短而获得虚高的分数。
            if chunks and len(tail) < self.chunk_size * 0.2:
                chunks[-1] = (chunks[-1] + tail)[: self.chunk_size * 2]
            else:
                chunks.append(tail)
        return chunks

    def _take_overlap(self, buffer: Sequence[str]) -> list[str]:
        """从缓冲区尾部取出不超过 chunk_overlap 长度的完整片段作为重叠。"""
        if self.chunk_overlap <= 0:
            return []
        overlap: list[str] = []
        length = 0
        for piece in reversed(buffer):
            if length + len(piece) > self.chunk_overlap and overlap:
                break
            overlap.insert(0, piece)
            length += len(piece)
        return overlap

    @staticmethod
    def _extract_leading_heading(text: str) -> str | None:
        match = MARKDOWN_HEADING.search(text)
        if match and text.strip().startswith("#"):
            return match.group(2).strip()
        first_line = text.strip().split("\n", 1)[0]
        return first_line[:60] if len(first_line) <= 60 else None

    @staticmethod
    def _split_by_heading(text: str) -> list[tuple[str, str]]:
        """按 Markdown 标题分节，返回 [(标题, 正文), ...]。"""
        matches = list(MARKDOWN_HEADING.finditer(text))
        if not matches:
            return [("", text)]

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for index, match in enumerate(matches):
            heading = match.group(0).strip()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections.append((heading, text[start:end].strip()))
        return sections


__all__ = ["SEPARATORS", "TextChunker"]
