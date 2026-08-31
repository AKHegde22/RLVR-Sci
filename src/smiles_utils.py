"""SMILES parsing and completion text extraction for RLVR rewards."""

import re
from contextlib import contextmanager
from typing import Any, Optional

from rdkit import Chem
from rdkit import RDLogger

SMILES_TAG_PATTERN = re.compile(r"<smiles>\s*(.*?)\s*</smiles>", re.IGNORECASE | re.DOTALL)
TOKEN_SPLIT_PATTERN = re.compile(r"[\s,;]+")
# Tokens that contain SMILES syntax beyond plain letters (bonds, rings, branches, etc.)
SMILES_SYNTAX_PATTERN = re.compile(r"[#=\-\[\]()\\/0-9@+%.]")
PURE_ALPHA_PATTERN = re.compile(r"^[A-Za-z]+$")


@contextmanager
def suppress_rdkit_logs():
    """Silence RDKit parse warnings while probing invalid SMILES candidates."""
    RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")


def _looks_like_smiles_candidate(token: str) -> bool:
    """Fast filter to avoid parsing obvious non-SMILES prose tokens."""
    if not token:
        return False
    if len(token) == 1 and token.isalpha():
        return True
    if PURE_ALPHA_PATTERN.match(token) and len(token) > 3:
        # e.g. invalid, molecule, text, here, your
        return False
    if SMILES_SYNTAX_PATTERN.search(token):
        return True
    # Compact alphanumeric fragments: CCO, c1ccccc1, CC(=O)O after strip
    if len(token) <= 128 and re.fullmatch(r"[A-Za-z0-9]+", token):
        return True
    return False


def mol_from_smiles_quiet(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES without spamming RDKit errors for invalid probes."""
    with suppress_rdkit_logs():
        return Chem.MolFromSmiles(smiles)


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
    mol = mol_from_smiles_quiet(smiles)
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
        candidate = tag_match.group(1).strip()
        if _looks_like_smiles_candidate(candidate):
            canonical = _canonicalize_smiles(candidate)
            if canonical is not None:
                return canonical

    if _looks_like_smiles_candidate(text):
        canonical = _canonicalize_smiles(text)
        if canonical is not None:
            return canonical

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line or not _looks_like_smiles_candidate(line):
            continue
        canonical = _canonicalize_smiles(line)
        if canonical is not None:
            return canonical

    for token in reversed(TOKEN_SPLIT_PATTERN.split(text)):
        token = token.strip("\"'`()[]{}")
        if not _looks_like_smiles_candidate(token):
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
