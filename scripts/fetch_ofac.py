#!/usr/bin/env python3
"""
Scripts A: Fetch OFAC SDN sanctions list and seed into KnowledgeBase.

Source: https://www.treasury.gov/ofac/downloads/sdn.xml (public domain, free)
Updated: daily by US Treasury

Usage:
    python scripts/fetch_ofac.py
    python scripts/fetch_ofac.py --limit 500  # only first N entries
    python scripts/fetch_ofac.py --dry-run    # parse only, don't store
"""

import asyncio
import argparse
import sys
import os
import xml.etree.ElementTree as ET
from typing import List, Dict
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_ofac")

OFAC_URL = "https://www.treasury.gov/ofac/downloads/sdn.xml"
UN_URL = "https://scsanctions.un.org/resources/xml/en/consolidated.xml"


def fetch_ofac_xml(url: str) -> str:
    """Download XML from URL."""
    try:
        import urllib.request
        logger.info(f"Downloading: {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def parse_ofac_sdn(xml_content: str, limit: int = 1000) -> List[Dict]:
    """
    Parse OFAC SDN XML into structured records.
    Returns list of dicts with name, type, programs, remarks.
    """
    records = []
    try:
        root = ET.fromstring(xml_content)
        ns = {"ofac": "http://tempuri.org/sdnList.xsd"}

        # Try with namespace first, then without
        entries = root.findall(".//sdnEntry") or root.findall(".//ofac:sdnEntry", ns)
        logger.info(f"Found {len(entries)} SDN entries")

        for entry in entries[:limit]:
            def get_text(tag):
                el = entry.find(tag) or entry.find(f"ofac:{tag}", ns)
                return el.text.strip() if el is not None and el.text else ""

            last_name = get_text("lastName")
            first_name = get_text("firstName")
            name = f"{first_name} {last_name}".strip() or last_name
            sdn_type = get_text("sdnType")
            title = get_text("title")
            remarks = get_text("remarks")

            # Programs
            programs = []
            for prog in (entry.findall(".//program") or entry.findall(".//ofac:program", ns)):
                if prog.text:
                    programs.append(prog.text.strip())

            if name:
                records.append({
                    "name": name,
                    "type": sdn_type,
                    "title": title,
                    "programs": programs,
                    "remarks": remarks[:200] if remarks else "",
                })
    except Exception as e:
        logger.error(f"XML parse error: {e}")

    return records


def records_to_knowledge_docs(records: List[Dict], batch_size: int = 50) -> List[Dict]:
    """
    Convert SDN records into knowledge base documents.
    Groups records into batches for efficient storage.
    """
    docs = []

    # Summary document
    total = len(records)
    entity_types = {}
    all_programs = set()
    for r in records:
        t = r["type"] or "Unknown"
        entity_types[t] = entity_types.get(t, 0) + 1
        all_programs.update(r["programs"])

    summary = (
        f"OFAC SDN (Specially Designated Nationals) List Summary:\n"
        f"Total sanctioned entities: {total}\n"
        f"Entity types: {', '.join(f'{k}({v})' for k,v in entity_types.items())}\n"
        f"Sanction programs: {', '.join(sorted(all_programs)[:30])}\n"
        f"Source: US Treasury OFAC. Updated daily.\n"
        f"All transactions with these entities are prohibited under US law."
    )
    docs.append({"content": summary, "category": "aml_regulations",
                 "metadata": {"source": "ofac_sdn", "doc_type": "summary"}})

    # Individual name batches
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        lines = []
        for r in batch:
            prog_str = ", ".join(r["programs"]) if r["programs"] else "GENERAL"
            line = f"- {r['name']} [{r['type']}] Programs: {prog_str}"
            if r["title"]:
                line += f" Title: {r['title']}"
            lines.append(line)

        content = (
            f"OFAC Sanctioned Entities (batch {i//batch_size + 1}):\n"
            + "\n".join(lines)
            + "\nSource: US Treasury OFAC SDN List"
        )
        docs.append({
            "content": content,
            "category": "aml_regulations",
            "metadata": {"source": "ofac_sdn", "doc_type": "entity_batch",
                         "batch": i // batch_size}
        })

    return docs


async def store_docs(docs: List[Dict], dry_run: bool = False) -> int:
    """Store documents into GlobalMemorySystem KnowledgeBase."""
    if dry_run:
        logger.info(f"[DRY RUN] Would store {len(docs)} documents")
        for d in docs[:3]:
            print(f"  [{d['category']}] {d['content'][:120]}...")
        return len(docs)

    from aegean.memory.global_memory import GlobalMemorySystem
    memory = GlobalMemorySystem()
    count = 0
    for doc in docs:
        try:
            await memory.add_knowledge(
                content=doc["content"],
                category=doc["category"],
                metadata=doc["metadata"],
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to store doc: {e}")

    logger.info(f"Stored {count}/{len(docs)} documents")
    return count


async def main(limit: int = 1000, dry_run: bool = False):
    logger.info("=== OFAC SDN Knowledge Base Seeder ===")

    # Download
    xml_content = fetch_ofac_xml(OFAC_URL)
    logger.info(f"Downloaded {len(xml_content):,} bytes")

    # Parse
    records = parse_ofac_sdn(xml_content, limit=limit)
    logger.info(f"Parsed {len(records)} SDN entries")

    # Convert to docs
    docs = records_to_knowledge_docs(records)
    logger.info(f"Generated {len(docs)} knowledge documents")

    # Store
    count = await store_docs(docs, dry_run=dry_run)
    logger.info(f"Done. {count} documents ready for RAG retrieval.")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OFAC SDN list into knowledge base")
    parser.add_argument("--limit", type=int, default=1000, help="Max SDN entries to process")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't store")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit, dry_run=args.dry_run))

