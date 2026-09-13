from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = '0001_initial_pgvector'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.create_table(
        'organizations',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column('slug', sa.String(length=100), nullable=False, unique=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        'workspaces',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('slug', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('index_version', sa.Integer(), server_default='1', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('organization_id', 'slug', name='uq_workspace_org_slug'),
    )
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column('email', sa.String(length=320), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(length=512), nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        'memberships',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(length=32), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint('user_id', 'workspace_id', name='uq_membership_user_workspace'),
    )
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_by_user_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('users.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('source_name', sa.String(length=255), nullable=False),
        sa.Column('media_type', sa.String(length=128), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('content_text', sa.Text(), nullable=False),
        sa.Column('document_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('error_code', sa.String(length=64), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_documents_workspace_status', 'documents', ['workspace_id', 'status'])
    op.create_index('ix_documents_workspace_sha', 'documents', ['workspace_id', 'sha256'])
    op.create_table(
        'document_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False),
        sa.Column('embedding', Vector(dim=16), nullable=False),
        sa.Column('embedding_model', sa.String(length=128), nullable=False),
        sa.Column('chunk_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_document_chunks_workspace_document', 'document_chunks', ['workspace_id', 'document_id'])
    op.execute(
        'CREATE INDEX ix_document_chunks_embedding ON document_chunks USING ivfflat '
        '(embedding vector_cosine_ops) WITH (lists = 100)'
    )
    op.execute(
        "CREATE INDEX ix_document_chunks_fts ON document_chunks USING gin "
        "(to_tsvector('english', content))"
    )
    op.create_table(
        'ingestion_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column('workspace_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False),
        sa.Column('requested_by_user_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('users.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=False), sa.ForeignKey('documents.id', ondelete='SET NULL'), nullable=True),
        sa.Column('operation', sa.String(length=32), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('attempt_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('error_code', sa.String(length=64), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_ingestion_jobs_workspace_status', 'ingestion_jobs', ['workspace_id', 'status'])


def downgrade() -> None:
    op.drop_index('ix_ingestion_jobs_workspace_status', table_name='ingestion_jobs')
    op.drop_table('ingestion_jobs')
    op.execute('DROP INDEX IF EXISTS ix_document_chunks_fts')
    op.execute('DROP INDEX IF EXISTS ix_document_chunks_embedding')
    op.drop_index('ix_document_chunks_workspace_document', table_name='document_chunks')
    op.drop_table('document_chunks')
    op.drop_index('ix_documents_workspace_sha', table_name='documents')
    op.drop_index('ix_documents_workspace_status', table_name='documents')
    op.drop_table('documents')
    op.drop_table('memberships')
    op.drop_table('users')
    op.drop_table('workspaces')
    op.drop_table('organizations')
