"""Enforce one active document per content hash; keyset-paging index; workspace chunk counts.

The old non-unique (workspace_id, sha256) index let two concurrent uploads of the same file
both pass the application's duplicate check. The partial unique index makes the database
reject the second one. Failed and deleted documents are excluded, so a failed upload can be
retried and a deleted document re-added.

``workspaces.chunk_count`` is backfilled here and then maintained in the same transactions
that add or remove chunks; vector search uses it to choose between exact and HNSW search.

If an existing database already holds duplicate active documents, creating the unique index
fails with a clear error; resolve the duplicates (delete one of each pair) and re-run.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_document_integrity_indexes"
down_revision = "0001_initial_pgvector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE UNIQUE INDEX uq_documents_workspace_sha_active ON documents "
        "(workspace_id, sha256) WHERE status NOT IN ('deleted', 'failed')"
    )
    op.drop_index("ix_documents_workspace_sha", table_name="documents")
    op.create_index(
        "ix_documents_workspace_created_id", "documents", ["workspace_id", "created_at", "id"]
    )
    op.add_column(
        "workspaces",
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.execute(
        "UPDATE workspaces w SET chunk_count = "
        "(SELECT count(*) FROM document_chunks c WHERE c.workspace_id = w.id)"
    )


def downgrade() -> None:
    op.drop_column("workspaces", "chunk_count")
    op.drop_index("ix_documents_workspace_created_id", table_name="documents")
    op.create_index("ix_documents_workspace_sha", "documents", ["workspace_id", "sha256"])
    op.execute("DROP INDEX IF EXISTS uq_documents_workspace_sha_active")
