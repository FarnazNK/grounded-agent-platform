variable "environment" { type = string }
variable "aws_region" { type = string }
variable "vpc_cidr" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "db_instance_class" { type = string }
variable "db_username" { type = string }
variable "db_password" { type = string, sensitive = true }
