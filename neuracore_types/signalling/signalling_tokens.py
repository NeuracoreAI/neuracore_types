"""Signalling session token request and response models."""

from datetime import datetime

from pydantic import BaseModel, Field


class SignallingTokenRequest(BaseModel):
    """Request for a signalling session token.

    Attributes:
        robot_id: Robot the token grants access to. If None, the token covers
            the organization.
    """

    robot_id: str | None = Field(default=None, min_length=1)


class SignallingTokenResponse(BaseModel):
    """Signed signalling session token.

    Attributes:
        token: Signed token to present to a signalling region.
        expires_at: Time after which regions reject the token.
        renew_after: Time after which the client should request a new token.
    """

    token: str
    expires_at: datetime
    renew_after: datetime
