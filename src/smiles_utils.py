"""SMILES parsing and completion text extraction for RLVR rewards."""

import re
from typing import Any, Optional

from rdkit import Chem

SMILES_TAG_PATTERN = re.compile(r"<smiles>\s*(.*?)\s*</smiles>", re.IGNORECASE | re.DOTALL)
TOKEN_SPLIT_PATTERN = re.compile(r"[\s,;]+")


def extract_completion_text(completion: Any) -> str:
    """Extract assistant text from TRL completion payloads (string or chat messages)."""
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content", "")
                if content:
                    return str(content)
        for message in completion:
            if isinstance(message, dict) and "content" in message:
                content = message.get("content", "")
                if content:
                    return str(content)
        return ""
    return str(completion)


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def extract_smiles(text: str) -> Optional[str]:
    """
    Extract and canonicalize a SMILES string from model output.
    Supports <smiles> tags, full-string parse, line scan, and token scan.
    """
    text = text.strip()
    if not text:
        return None

    tag_match = SMILES_TAG_PATTERN.search(text)
    if tag_match:
        canonical = _canonicalize_smiles(tag_match.group(1).strip())
        if canonical is not None:
            return canonical

    canonical = _canonicalize_smiles(text)
    if canonical is not None:
        return canonical

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        canonical = _canonicalize_smiles(line)
        if canonical is not None:
            return canonical

    for token in reversed(TOKEN_SPLIT_PATTERN.split(text)):
        token = token.strip("\"'`()[]{}")
        if not token:
            continue
        canonical = _canonicalize_smiles(token)
        if canonical is not None:
            return canonical

    return None


def extract_smiles_from_completion(completion: Any) -> Optional[str]:
    """Parse SMILES from a TRL completion object."""
    return extract_smiles(extract_completion_text(completion))


def prompt_to_key(prompt: Any) -> str:
    """Stable hashable key for grouping GRPO generations that share a prompt."""
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, dict):
                parts.append(f"{message.get('role', '')}:{message.get('content', '')}")
        return "|".join(parts)
    return str(prompt)
