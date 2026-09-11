import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Matches a markdown table: header row + separator row + one or more data rows
TABLE_PATTERN = re.compile(r"(\|.+\|\n\|[-:| ]+\|\n(?:\|.+\|\n?)+)", re.MULTILINE)


def read_markdown(markdown_file: str) -> str:
    """Read markdown content"""
    return Path(markdown_file).read_text(encoding="utf-8")


def extract_tables(markdown_text: str) -> tuple[str, list[str]]:
    """
    Pulls markdown tables out of the text and replaces them with a
    placeholder, so the header/size splitters never cut a table in half.
    """
    tables = TABLE_PATTERN.findall(markdown_text)
    text_without_tables = TABLE_PATTERN.sub("\n[TABLE_PLACEHOLDER]\n", markdown_text)
    return text_without_tables, tables


def is_valid_chunk(chunk: Document, min_length: int = 100) -> bool:
    """
    Reject chunks that are too short or look like letterhead / AGM notice /
    signature-block boilerplate rather than actual financial content.
    """
    text = chunk.page_content.strip()

    if len(text) < min_length:
        return False

    noise_patterns = [
        r"shareholder\.grievances",
        r"digitally signed",
        r"notice is hereby given",
        r"annual general meeting",
        r"tel\.\s*no",
        r"^sub:",
        r"yours faithfully",
        r"for .*(bank|limited|ltd)\.? *$",
    ]
    lower_text = text.lower()
    if any(re.search(pattern, lower_text) for pattern in noise_patterns):
        return False

    contact_indicators = ["www.", "@", "tel.", "fax"]
    lines = text.split("\n")
    contact_line_count = sum(
        1 for line in lines if any(ind in line.lower() for ind in contact_indicators)
    )
    if contact_line_count / max(len(lines), 1) > 0.4:
        return False

    return True


def chunk_markdown(markdown_file: str) -> list[Document]:
    """
    Generate structure-aware chunks from markdown (splits on headers,
    keeps tables intact, further splits any oversized section)
    """
    markdown_content = read_markdown(markdown_file)

    text_without_tables, tables = extract_tables(markdown_content)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    section_docs = header_splitter.split_text(text_without_tables)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    sized_docs = size_splitter.split_documents(section_docs)

    final_docs: list[Document] = []
    table_queue = list(tables)

    for doc in sized_docs:
        if "[TABLE_PLACEHOLDER]" in doc.page_content:
            cleaned_text = doc.page_content.replace("[TABLE_PLACEHOLDER]", "").strip()
            if cleaned_text:
                final_docs.append(Document(page_content=cleaned_text, metadata=doc.metadata))
            if table_queue:
                table_text = table_queue.pop(0)
                final_docs.append(Document(
                    page_content=table_text,
                    metadata={**doc.metadata, "chunk_type": "table"},
                ))
        else:
            final_docs.append(doc)

    for table_text in table_queue:
        final_docs.append(Document(page_content=table_text, metadata={"chunk_type": "table"}))

    return final_docs


def chunk_directory(markdown_dir: str) -> list[Document]:
    """
    Generate structure-aware, filtered chunks from all markdown files
    in a directory
    """
    all_chunks = []
    files = sorted(Path(markdown_dir).glob("*.md"))
    for i, md_file in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing {md_file.name} ...")
        chunks = chunk_markdown(str(md_file))
        for chunk in chunks:
            chunk.metadata["source"] = md_file.name

        chunks = [c for c in chunks if is_valid_chunk(c)]

        all_chunks.extend(chunks)
        logger.info(f"  -> {len(chunks)} valid chunks generated")
    return all_chunks


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    default_path = repo_root / "data" / "markdown"

    chunks = chunk_directory(str(default_path))

    logger.info(f"Total valid chunks: {len(chunks)}")
    for c in chunks[:5]:
        print("---")
        print(c.metadata, "|", c.page_content[:150])