from typing import TYPE_CHECKING

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from STIM.models.base import DBModel, UUIDMixin

if TYPE_CHECKING:
    from STIM.models.conversation.model import ConversationModel


class ConnectionModel(DBModel, UUIDMixin, kw_only=True):
    __tablename__ = "connections"
    dsn: Mapped[str] = mapped_column("dsn", String, nullable=False, unique=True)
    database: Mapped[str] = mapped_column("database", String, nullable=False)
    name: Mapped[str | None] = mapped_column("name", String)
    type: Mapped[str] = mapped_column("type", String, nullable=False)
    dialect: Mapped[str | None] = mapped_column("dialect", String)
    is_sample: Mapped[bool] = mapped_column("is_sample", Boolean, nullable=False, default=False, server_default="false")

    # Relationships
    conversations: Mapped[list["ConversationModel"]] = relationship("ConversationModel", back_populates="connection")
