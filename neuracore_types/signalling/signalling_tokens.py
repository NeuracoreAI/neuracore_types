"""Signalling session token request and response models."""

from datetime import datetime

from pydantic import BaseModel, Field


class SignallingTokenRequest(BaseModel):
    """Request for a signalling session token.

    Attributes:
        robot_id: Robot the token grants access to. If None, the token covers
            the organization.
        robot_instance: Robot instance of a robot token.
        register_epoch: Sign a new registration epoch into the token: the
            issue time in microseconds since the Unix epoch.
    """

    robot_id: str | None = Field(default=None, min_length=1)
    robot_instance: int | None = Field(default=None, ge=0)
    register_epoch: bool = False


class SignallingTokenResponse(BaseModel):
    """Signed signalling session token.

    Attributes:
        token: Signed token to present to a signalling region.
        expires_at: Time after which regions reject the token.
        renew_after: Time after which the client should request a new token.
        region_endpoints: Canonical URLs of every signalling region, for an
            organization token.
    """

    token: str
    expires_at: datetime
    renew_after: datetime
    region_endpoints: list[str] = Field(default_factory=list)
