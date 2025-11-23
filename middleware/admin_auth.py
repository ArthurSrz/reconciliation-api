"""
Admin authentication middleware for protected endpoints.

This module provides authentication for admin-only endpoints like book ingestion.
Uses X-Admin-API-Key header for API authentication.

Constitution Compliance:
- Principle VI: Extensible Literature Foundation - admin-only access protects graph integrity
"""

import os
from functools import wraps
from typing import Callable, Optional
from flask import request, jsonify, g
import hashlib
import hmac


# Environment variable for admin API key
ADMIN_API_KEY_ENV = "ADMIN_API_KEY"

# Default admin ID when using CLI (not API)
CLI_ADMIN_ID = "cli-admin"


def get_admin_api_key() -> Optional[str]:
    """Get the admin API key from environment."""
    return os.environ.get(ADMIN_API_KEY_ENV)


def verify_admin_key(provided_key: str) -> bool:
    """
    Verify the provided admin API key against the configured key.

    Uses constant-time comparison to prevent timing attacks.
    """
    expected_key = get_admin_api_key()
    if not expected_key:
        # If no key is configured, reject all requests
        return False

    # Use hmac.compare_digest for constant-time comparison
    return hmac.compare_digest(provided_key, expected_key)


def get_admin_id() -> str:
    """
    Get the admin ID from the current request context.

    Returns CLI_ADMIN_ID if not in a request context.
    """
    try:
        return getattr(g, 'admin_id', CLI_ADMIN_ID)
    except RuntimeError:
        # Outside of request context (e.g., CLI usage)
        return CLI_ADMIN_ID


def require_admin_key(f: Callable) -> Callable:
    """
    Decorator to require admin authentication for endpoints.

    Usage:
        @app.route('/admin/ingest', methods=['POST'])
        @require_admin_key
        def ingest_book():
            admin_id = get_admin_id()
            ...

    The decorator checks for X-Admin-API-Key header and verifies it.
    Sets g.admin_id for use in the decorated function.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if admin key is configured
        if not get_admin_api_key():
            return jsonify({
                "error": "admin_not_configured",
                "message": "Admin API key is not configured on the server. "
                           "Set ADMIN_API_KEY environment variable."
            }), 500

        # Get the provided API key from header
        provided_key = request.headers.get('X-Admin-API-Key')

        if not provided_key:
            return jsonify({
                "error": "unauthorized",
                "message": "Admin authentication required. "
                           "Provide X-Admin-API-Key header."
            }), 401

        # Verify the key
        if not verify_admin_key(provided_key):
            return jsonify({
                "error": "forbidden",
                "message": "Invalid admin API key."
            }), 403

        # Set admin ID based on a hash of the key (for audit purposes)
        # This allows tracking which key was used without storing the key itself
        g.admin_id = f"admin-{hashlib.sha256(provided_key.encode()).hexdigest()[:8]}"

        return f(*args, **kwargs)

    return decorated_function


def require_admin_key_or_cli(f: Callable) -> Callable:
    """
    Decorator that allows both API key authentication and CLI usage.

    For CLI usage, set the environment variable ADMIN_CLI_MODE=true
    to bypass API key requirement.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if in CLI mode
        if os.environ.get('ADMIN_CLI_MODE', '').lower() == 'true':
            g.admin_id = CLI_ADMIN_ID
            return f(*args, **kwargs)

        # Otherwise, require admin key
        return require_admin_key(f)(*args, **kwargs)

    return decorated_function


class AdminAuthError(Exception):
    """Exception raised for admin authentication errors."""

    def __init__(self, message: str, error_code: str = "auth_error"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


def validate_cli_admin() -> str:
    """
    Validate admin access for CLI operations.

    For CLI operations, we check if ADMIN_API_KEY is set.
    Returns the admin ID to use for the operation.

    Raises:
        AdminAuthError: If admin key is not configured
    """
    if not get_admin_api_key():
        raise AdminAuthError(
            "Admin API key not configured. Set ADMIN_API_KEY environment variable.",
            "admin_not_configured"
        )
    return CLI_ADMIN_ID
