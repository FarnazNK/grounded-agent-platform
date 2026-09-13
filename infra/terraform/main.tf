terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_vpc" "rag_agent" {
  cidr_block = var.vpc_cidr
  tags = { Name = "rag-agent-${var.environment}" }
}

resource "aws_db_subnet_group" "rag_agent" {
  name       = "rag-agent-${var.environment}"
  subnet_ids = var.private_subnet_ids
}

resource "aws_security_group" "api" {
  name   = "rag-agent-api-${var.environment}"
  vpc_id = aws_vpc.rag_agent.id
}

resource "aws_db_instance" "postgres" {
  identifier             = "rag-agent-${var.environment}"
  engine                 = "postgres"
  instance_class         = var.db_instance_class
  allocated_storage      = 20
  username               = var.db_username
  password               = var.db_password
  db_name                = "rag_agent"
  skip_final_snapshot    = true
  db_subnet_group_name   = aws_db_subnet_group.rag_agent.name
  vpc_security_group_ids = [aws_security_group.api.id]
}

resource "aws_ecs_cluster" "rag_agent" {
  name = "rag-agent-${var.environment}"
}
