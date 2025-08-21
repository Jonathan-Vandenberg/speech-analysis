"""
Encryption utilities for securely storing per-tenant secrets in the control plane.

Uses Fernet (AES-128 in CBC with HMAC) via the cryptography library. The key is
derived from the environment variable `CP_ENCRYPTION_KEY`. You can set it either
as a 32-byte base64 urlsafe key (recommended) or provide a passphrase via
`CP_ENCRYPTION_PASSPHRASE`, which will be stretched with PBKDF2-HMAC-SHA256.

This module exposes two functions:
 - encrypt_string(plaintext: str) -> str
 - decrypt_string(ciphertext: str) -> str

The returned ciphertext is a urlsafe base64 string that includes an authentication
tag and timestamp, making it safe to store in text columns.
"""
from __future__ import annotations

import base64
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def _derive_key_from_passphrase(passphrase: str, salt: bytes) -> bytes:
    """Derive a Fernet key from a passphrase using PBKDF2-HMAC-SHA256.

    The returned key is base64 urlsafe-encoded 32 bytes, suitable for Fernet.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
    )
    key = kdf.derive(passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(key)


def _load_fernet() -> Fernet:
    """Load a Fernet instance from env.

    Precedence:
      1) CP_ENCRYPTION_KEY (base64 urlsafe 32 bytes)
      2) CP_ENCRYPTION_PASSPHRASE + CP_ENCRYPTION_SALT (base64) or default static salt
    """
    raw_key: Optional[str] = os.getenv("CP_ENCRYPTION_KEY")
    if raw_key:
        key_bytes = raw_key.encode("utf-8")
        # Validate length by attempting to construct Fernet
        return Fernet(key_bytes)

    passphrase = os.getenv("CP_ENCRYPTION_PASSPHRASE")
    if not passphrase:
        raise RuntimeError(
            "Missing CP_ENCRYPTION_KEY or CP_ENCRYPTION_PASSPHRASE for control plane secret encryption"
        )

    salt_b64 = os.getenv("CP_ENCRYPTION_SALT")
    if salt_b64:
        salt = base64.urlsafe_b64decode(salt_b64)
    else:
        # Static, non-secret salt is acceptable when passphrase is strong and rotated; for production
        # provide CP_ENCRYPTION_SALT to ensure uniqueness per environment.
        salt = b"school-ai-control-plane-salt"

    derived = _derive_key_from_passphrase(passphrase, salt)
    return Fernet(derived)


def encrypt_string(plaintext: str) -> str:
    """Encrypt a plaintext string using the configured Fernet key."""
    if plaintext is None:
        raise ValueError("plaintext must not be None")
    f = _load_fernet()
    token = f.encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_string(ciphertext: str) -> str:
    """Decrypt a ciphertext string using the configured Fernet key."""
    if ciphertext is None:
        raise ValueError("ciphertext must not be None")
    f = _load_fernet()
    try:
        data = f.decrypt(ciphertext.encode("utf-8"))
        return data.decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Invalid encryption key or corrupted ciphertext") from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encrypt/decrypt helper for control plane secrets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encrypt")
    enc.add_argument("text", help="Plaintext to encrypt")

    dec = sub.add_parser("decrypt")
    dec.add_argument("token", help="Ciphertext token to decrypt")

    args = parser.parse_args()
    if args.cmd == "encrypt":
        print(encrypt_string(args.text))
    elif args.cmd == "decrypt":
        print(decrypt_string(args.token))


