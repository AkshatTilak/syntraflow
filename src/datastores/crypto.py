"""Crypto and masking utilities for datastore binding credentials and connection URIs (S6-04b)."""

import json
import logging
import re
from typing import Any, Dict, Optional
import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from common.config.settings import get_settings

logger = logging.getLogger("syntraflow.datastores.crypto")

_URI_PASSWORD_RE = re.compile(r"://([^:@]+):([^@]+)@")


def get_fernet_key() -> str:
    """Get Fernet key from settings or derive dev fallback key."""
    settings = get_settings()
    key = settings.DATASTORE_ENCRYPTION_KEY
    if not key:
        # Generate deterministic dev fallback key from JWT_SECRET_KEY if empty
        raw = (settings.JWT_SECRET_KEY + "_datastore_key_salt").encode("utf-8")
        key = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode("utf-8")
    else:
        # Ensure it is valid base64 key or pad/hash if non-conforming
        try:
            Fernet(key.encode("utf-8") if isinstance(key, str) else key)
        except Exception:
            raw = key.encode("utf-8")
            key = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode("utf-8")
    return key


def _fernet() -> Fernet:
    """Instantiate Fernet instance with key."""
    key = get_fernet_key()
    return Fernet(key.encode("utf-8") if isinstance(key, str) else key)


def encrypt_credentials(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """Encrypt credentials dictionary to Fernet token string. Returns None if payload is empty/None."""
    if not payload:
        return None
    try:
        raw_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        f = _fernet()
        return f.encrypt(raw_json).decode("utf-8")
    except Exception as e:
        logger.error("Failed to encrypt credentials: %s", e)
        raise ValueError(f"Credential encryption failed: {e}") from e


def decrypt_credentials(blob: Optional[str]) -> Dict[str, Any]:
    """Decrypt Fernet blob to credentials dictionary. Returns empty dict if blob is None/empty."""
    if not blob:
        return {}
    try:
        f = _fernet()
        decrypted_bytes = f.decrypt(blob.encode("utf-8"))
        return json.loads(decrypted_bytes.decode("utf-8"))
    except InvalidToken as e:
        logger.error("Invalid token when decrypting credentials: %s", e)
        raise ValueError("Decryption failed: invalid encryption token") from e
    except Exception as e:
        logger.error("Failed to decrypt credentials: %s", e)
        raise ValueError(f"Credential decryption failed: {e}") from e


def mask_uri(uri: str) -> str:
    """Mask connection password in URI string.

    Example: postgresql://user:secret@host:5432/db -> postgresql://user:***@host:5432/db
    """
    if not uri:
        return ""
    return _URI_PASSWORD_RE.sub(r"://\1:***@", uri)


async def verify_encryption_key_configured(session: AsyncSession) -> None:
    """Raise RuntimeError at startup if DATASTORE_ENCRYPTION_KEY is missing and binding rows exist."""
    from common.models.database import DatastoreBinding

    settings = get_settings()
    if not settings.DATASTORE_ENCRYPTION_KEY:
        stmt = select(func.count(DatastoreBinding.id))
        res = await session.execute(stmt)
        count = res.scalar() or 0
        if count > 0:
            raise RuntimeError(
                "DATASTORE_ENCRYPTION_KEY is missing but datastore_bindings rows exist. "
                "Plaintext credential storage is strictly prohibited."
            )
