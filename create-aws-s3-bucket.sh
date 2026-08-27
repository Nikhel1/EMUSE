#!/usr/bin/env bash
#
# create-aws-s3-bucket.sh
#
# Creates an S3 bucket in ap-southeast-2 using the "Boss" AWS CLI profile,
# with sensible defaults (private, versioned, encrypted, public access blocked).
#
# Usage:
#   chmod +x create-aws-s3-bucket.sh
#   ./create-aws-s3-bucket.sh emu-data-bucket-2026
#
# Optional flags:
#   --region <region>    default: ap-southeast-2
#   --profile <profile>  default: Boss
#   --public             allow public read access (off by default)

set -euo pipefail

REGION="ap-southeast-2"
PROFILE="Boss"
PUBLIC=false
BUCKET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --public) PUBLIC=true; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^#//'
      exit 0
      ;;
    *) BUCKET="$1"; shift ;;
  esac
done

[[ -n "$BUCKET" ]] || { echo "Usage: $0 [--region <region>] [--profile <profile>] [--public] <bucket-name>" >&2; exit 1; }

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
ok()   { printf '\033[1;32m✔ %s\033[0m\n' "$1"; }

# ---------- 1. create the bucket ----------
log "Creating bucket '$BUCKET' in $REGION (profile: $PROFILE)"

# S3 bucket names are globally unique across ALL AWS accounts, not just yours.
# ap-southeast-2 needs a LocationConstraint (unlike us-east-1).
aws s3api create-bucket \
  --bucket "$BUCKET" \
  --region "$REGION" \
  --create-bucket-configuration LocationConstraint="$REGION" \
  --profile "$PROFILE"

ok "Bucket created"

# ---------- 2. block public access (unless --public was passed) ----------
if ! $PUBLIC; then
  log "Blocking public access (default, recommended)"
  aws s3api put-public-access-block \
    --bucket "$BUCKET" \
    --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true \
    --profile "$PROFILE"
  ok "Public access blocked"
else
  log "Skipping public access block (--public was set) — bucket policy still required to actually allow public reads"
fi

# ---------- 3. enable versioning ----------
log "Enabling versioning"
aws s3api put-bucket-versioning \
  --bucket "$BUCKET" \
  --versioning-configuration Status=Enabled \
  --profile "$PROFILE"
ok "Versioning enabled"

# ---------- 4. enable default encryption (SSE-S3) ----------
log "Enabling default encryption (AES256)"
aws s3api put-bucket-encryption \
  --bucket "$BUCKET" \
  --server-side-encryption-configuration '{
    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
  }' \
  --profile "$PROFILE"
ok "Default encryption enabled"

log "Done. Bucket URL: s3://$BUCKET"
echo "  Console: https://s3.console.aws.amazon.com/s3/buckets/$BUCKET?region=$REGION"
