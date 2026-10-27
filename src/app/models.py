"""Validated data models shared across the assistant application."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentChunk(BaseModel):
    """A passage extracted from a source document."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=20_000)
    source: str = Field(min_length=1, max_length=1024)
    page: int | None = Field(default=None, ge=1)
    embedding: list[float] | None = Field(default=None, min_length=1)

    @field_validator("id", "chunk_id", "content", "source", mode="before")
    @classmethod
    def strip_required_strings(cls, value: object) -> object:
        """Strip surrounding whitespace before validating strings."""
        if isinstance(value, str):
            return value.strip()
        return value


class Citation(BaseModel):
    """A numbered reference to a supporting source passage."""

    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)
    source: str = Field(min_length=1, max_length=1024)
    page: int | None = Field(default=None, ge=1)
    chunk_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("source", "chunk_id", "text", mode="before")
    @classmethod
    def strip_required_strings(cls, value: object) -> object:
        """Strip surrounding whitespace before validating strings."""
        if isinstance(value, str):
            return value.strip()
        return value


class ChatRequest(BaseModel):
    """The user's chat question and retrieval preferences."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("session_id", "question", mode="before")
    @classmethod
    def strip_required_strings(cls, value: object) -> object:
        """Strip surrounding whitespace before validating strings."""
        if isinstance(value, str):
            return value.strip()
        return value


class ChatResponse(BaseModel):
    """The answer and supporting metadata returned to the client."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    answer: str = Field(min_length=1, max_length=12_000)
    citations: list[Citation] = Field(default_factory=list)
    pii_detected: list[str] = Field(default_factory=list)

    @field_validator("session_id", "answer", mode="before")
    @classmethod
    def strip_required_strings(cls, value: object) -> object:
        """Strip surrounding whitespace before validating strings."""
        if isinstance(value, str):
            return value.strip()
        return value
