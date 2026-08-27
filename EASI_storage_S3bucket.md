# Setting Up an AWS S3 Bucket for Use with EASI

## What this document is about

EASI (the Earth Analytics Science and Innovation platform, https://research.csiro.au/easi/) is CSIRO's shared cloud environment for running large-scale data analysis — JupyterHub notebooks, batch processing jobs, and custom applications. Under the hood, EASI runs on **Kubernetes** (often shortened to "k8s"), which is a system for automatically deploying, scaling, and managing containerised applications across a cluster of machines — it's what lets many different users' notebooks and apps run side-by-side on shared infrastructure without interfering with each other.

One thing that catches people out when they move from a laptop or an HPC cluster to a cloud platform like EASI is storage. On a laptop or an HPC system (e.g. Pawsey, Virga), there's a normal filesystem: folders and files on a disk that any program can just open, read, and write. Cloud platforms generally don't give you that. Instead, large-scale data in the cloud lives in **object storage** — on AWS this is called **S3 (Simple Storage Service)**. An S3 "bucket" is a named container that holds files ("objects"), accessed over the network via an API rather than mounted as a disk. This is a different model, but every major analysis library (`xarray`, `astropy`, `dask`, `rioxarray`, etc.) knows how to read data straight out of S3, so in practice it works almost as smoothly as reading local files.

For working with EASI, there are two common ways to get data into S3-backed storage:

- **Use the built-in scratch bucket**, `s3://easihub-csiro-user-scratch/`, which is available to everyone on EASI out of the box. This is fine for temporary or working data, but it's shared, scratch-tier space rather than storage under one's own control.
- **Use an external AWS account and one's own bucket.** This takes a bit more setup, but gives full control over storage class, lifecycle rules, cost, and — importantly — exactly who and what is allowed to read or write the data. This is generally the better option for larger datasets or anything meant to stick around long-term.

This document walks through the second option end to end: creating an AWS account, installing the AWS command-line tool, creating a personal S3 bucket, uploading data to it, checking what it costs, and then granting EASI's Kubernetes-based services permission to read and write it, so notebooks and apps running on EASI can pull the data straight in.

---

## 1. Create an AWS account and an IAM user

### Sign up for an AWS account

Go to https://aws.amazon.com and sign up with an email address and a payment method (a card is required even if usage stays within the free tier). This creates a brand-new AWS account with one special identity already in it: the **root user**.

### Understand the root user

The root user is tied directly to the email address used to sign up, and it has completely unrestricted access to everything in the account — every resource, every setting, and the billing/payment details. Because of that unrestricted power, AWS's own guidance (and general good practice) is to **not use the root user for everyday work**. Instead:

- Turn on multi-factor authentication (MFA) on the root user immediately.
- Use the root user only for the handful of things that genuinely require it (e.g. closing the account, changing the support plan), and otherwise leave it logged out.
- Create a separate, ordinary identity — an **IAM user** — for day-to-day work, and use that instead.

### Create an IAM user with full access, via a group

IAM (Identity and Access Management) is AWS's system for creating identities and controlling what each one is allowed to do. Rather than granting permissions to one person at a time, it's common practice to create a **group** (a named bundle of permissions) and then add individual users to it — that way, permissions are managed once, on the group, rather than repeated per user.

To set this up:

1. In the AWS Console, go to **IAM → User groups → Create group**. Give it a name (e.g. `Admins`).
2. While creating the group, attach the **`AdministratorAccess`** managed policy to it. This is an AWS-provided policy that grants full access to essentially all AWS services and resources in the account — equivalent, in practice, to what the root user can do, minus a few account-level settings like closing the account or changing root credentials.
3. Go to **IAM → Users → Create user**. Name the user `Boss` (or any preferred name — this is the identity used throughout the rest of this document).
4. Add the `Boss` user to the `Admins` group created above, so it inherits the `AdministratorAccess` permissions.
5. Finish creating the user. At this point `Boss` is a normal IAM identity with full permissions on the account, ready to be used with the AWS CLI in the next section — there's no need to log in as root again for routine work.

> Giving a user full `AdministratorAccess` is convenient for getting started (and is what the rest of this guide assumes), but on a shared or long-lived account it's worth narrowing this down later to only the permissions actually needed (e.g. just S3-related actions), following the security principle of least privilege.

---

## 2. Install the AWS CLI

The AWS CLI is a single self-contained installer. Run:

```bash
curl -fsSL 'https://awscli.amazonaws.com/v2/install.sh' | bash
```

This downloads the official AWS installer script and runs it immediately, installing AWS CLI v2 into the home directory — no administrator/root privileges required.

### Make `aws` available in the terminal

The installer places the `aws` binary somewhere like `~/.local/bin`, which may not yet be on the shell's search `PATH` (the list of folders a shell checks when looking for a command). Add it for the current session:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

That only lasts until the terminal is closed. To make it permanent, append the same line to the shell's startup file so it runs automatically every time a new terminal opens:

```bash
SHELL_RC="$HOME/.bashrc"
if [ "$(basename "$SHELL")" = "zsh" ]; then
  SHELL_RC="$HOME/.zshrc"
fi
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC" && source "$SHELL_RC"
```

This checks whether the default shell is `zsh` (common on modern macOS) or `bash` (common on Linux/HPC systems), picks the matching config file, appends the `export PATH=...` line to it, and reloads it (`source`) so the change applies immediately.

Confirm the install worked:

```bash
aws --version
```

This should print something like `aws-cli/2.x.x`.

---

## 3. Configure a profile and log in

The AWS CLI can store credentials for more than one identity at once, each under a name (a "profile"). The examples below use a profile called `Boss`, matching the IAM user created in Step 1 — any name works, as long as it's used consistently.

### Set a default region

```bash
aws configure set region ap-southeast-2 --profile Boss
```

AWS data centres are grouped into geographic **regions**; `ap-southeast-2` is Sydney. This creates (or updates) the `Boss` profile in `~/.aws/config` with that as its default region.

### Log in

```bash
aws login --region ap-southeast-2 --profile Boss
```

This opens a browser-based (or device-code) sign-in flow for the `Boss` IAM user and stores temporary, expiring credentials under the `Boss` profile — nothing permanent is typed in here.

On a remote machine with no browser (e.g. an HPC login node like CSIRO's **virga**), add `--remote`:

```bash
aws login --region ap-southeast-2 --profile Boss --remote
```

This prints a verification code and a URL. Open the URL on any device with a browser, enter the code, and approve the sign-in — the remote session then picks up valid credentials automatically.

### Verify the login worked

```bash
aws sts get-caller-identity --profile Boss
```

`sts` (Security Token Service) is AWS's identity-check API — this simply asks "who am I, according to AWS?" and returns an account ID, user/role identifier, and user ID. Valid-looking output means authentication succeeded.

---

## 4. (Optional) Set up the AI Agent Toolkit

For anyone using AI coding assistants (Claude Code, Cursor, Cline, Kiro, etc.) that support MCP (Model Context Protocol), AWS provides a one-line setup that lets the assistant query AWS directly:

```bash
aws configure agent-toolkit --yes --region us-east-1 --profile Boss
```

This installs default AWS-related tools for supported coding agents and configures an MCP server connection.

> The Agent Toolkit backend service currently only exists in `us-east-1`, regardless of which region data is actually stored in. Always use `us-east-1` for this specific command — the bucket itself doesn't need to live there.

The command edits one or more MCP configuration files, adding an entry like:

```json
"aws-mcp": {
  "command": "uvx",
  "args": ["mcp-proxy-for-aws@latest", "https://aws-mcp.us-east-1.api.aws/mcp", "--metadata", "INSTALL_SOURCE=aws-cli"],
  "timeout": 100000,
  "transport": "stdio"
}
```

By default this has no `env` block, meaning the MCP server uses the *default* AWS profile rather than a specific named one. To pin it to a chosen profile (e.g. `Boss`), add an `env` block, leaving everything else untouched:

```json
"aws-mcp": {
  "command": "uvx",
  "args": ["mcp-proxy-for-aws@latest", "https://aws-mcp.us-east-1.api.aws/mcp", "--metadata", "INSTALL_SOURCE=aws-cli"],
  "env": {
    "AWS_MCP_PROXY_PROFILES": "Boss"
  },
  "timeout": 100000,
  "transport": "stdio"
}
```

Which file to edit depends on the tool in use:

| Agent       | MCP configuration file       |
|-------------|-------------------------------|
| Claude Code | `~/.claude.json`               |
| Cline       | `~/.cline/mcp.json`             |
| Cursor      | `~/.cursor/mcp.json`            |
| Kiro        | `~/.kiro/settings/mcp.json`     |

This step is entirely optional and has no bearing on whether S3/EASI access works — skip it without an AI coding agent in use.

---

## 5. Create an S3 bucket

Log in again if the session has expired (temporary credentials from `aws login` typically expire after a few hours):

```bash
aws login --region ap-southeast-2 --profile Boss
```

**Avoiding repeated logins:** instead of running `aws login` every session, it's possible to generate a long-lived IAM access key pair for the `Boss` user (via the AWS Console, under IAM → Users → Security credentials) and export it in the shell startup file:

```bash
# AWS credentials for IAM user Boss
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
```

This is more convenient but less secure — these two values should be treated like a password: never committed to a git repository or shared. A short-lived login via `aws login` is the safer default; static access keys are only worth the trade-off if that risk is understood and accepted.

### Run the bucket-creation script

A ready-made helper script for this workflow is available here:
https://github.com/Nikhel1/EMUSE/blob/docker-bucket-data/create-aws-s3-bucket.sh

Download it and run it with the desired bucket name:

```bash
./create-aws-s3-bucket.sh emu-data-bucket-2026
```

S3 bucket names must be **globally unique across all of AWS** (not just one account), lowercase, and contain no spaces or underscores — including a project name and year (e.g. `emu-data-bucket-2026`) is a common way to keep them unique and identifiable.

---

## 6. Estimating and checking storage cost

Before or after uploading a large dataset, it's worth estimating the cost. As a worked example, for a bucket holding 367 FITS files totalling 291.7 GB in the `us-east-1` region, stored in the default **S3 Standard** storage class:

| Cost component | Rate | Monthly cost |
|---|---|---|
| Storage (S3 Standard) | $0.023/GB/month | ≈ $6.71/month |
| GET requests (moderate access) | $0.0004 per 1,000 requests | ≈ $0.004/month |
| PUT requests (one-time upload) | $0.005 per 1,000 requests | ≈ $0.002 (one-time) |
| Data transfer within AWS | $0.00 | $0.00 |

**Downloading data out of AWS** (e.g. to a laptop or an on-prem cluster) is billed separately and is usually the more expensive part:

| Scenario | Cost |
|---|---|
| First 100 GB out per month | Free |
| Remaining data beyond that (e.g. ~192 GB) | ≈ $17.25 one-time |
| Transfer to another AWS service in the *same* region (e.g. EASI/EC2) | $0.00 |

That last row matters for this workflow: reading a bucket from EASI, when EASI and the bucket are in compatible AWS regions/accounts, does not incur the "data transfer out" charge that downloading to a laptop would.

### Reducing cost for infrequently-accessed archives

Large FITS files that are uploaded once and read only occasionally are good candidates for a cheaper storage class:

- **S3 Intelligent-Tiering** — automatically monitors access patterns and moves objects to a cheaper tier once they go unused for a while, with no retrieval fee. Can cut storage cost by roughly 40–50% for archival-style data.
- **S3 Standard-IA** ("Infrequent Access") — cheaper per-GB storage (~$0.0125/GB/month) in exchange for a small per-GB retrieval fee and a 30-day minimum storage duration. Suits data accessed less than once a month.
- **S3 Glacier Instant Retrieval** — the cheapest option (~$0.004/GB/month) that still allows millisecond-speed retrieval, aimed at data accessed only a few times a year.

Check the actual size and object count of anything already uploaded with:

```bash
aws s3 ls s3://emu-data-bucket-2026/images/ --summarize
```

`--summarize` adds a total object count and total size at the end of the listing, which is otherwise just a flat file list.

---

## 7. Upload data

Once the bucket exists, copy a local directory of files into it:

```bash
aws s3 sync ./images s3://emu-data-bucket-2026/images/
```

`aws s3 sync` compares the source folder against the destination and only uploads files that are new or changed, which makes it safe to re-run if an upload gets interrupted, rather than re-uploading everything (which plain `aws s3 cp` would do).

Confirm what actually landed in the bucket:

```bash
aws s3 ls s3://emu-data-bucket-2026/images/ --summarize
```

---

## 8. Grant EASI's Kubernetes services permission to access the bucket

By default, an S3 bucket is private to the AWS account that owns it — nothing else, including anything running on EASI, can read or write to it. To allow EASI's services to access it, attach a **bucket policy**: a JSON document, attached directly to the bucket, that grants specific permissions to specific AWS identities.

EASI itself runs on Kubernetes, and different parts of it operate under different **IAM roles**. A role is similar to the IAM user created earlier, but instead of being logged into directly with a password or access keys, it's *assumed* by a service — meaning EASI's own infrastructure takes on that identity temporarily to perform an action, without a person ever entering credentials for it. Two relevant roles:

- `easihub-csiro-csiro-easihub-client` — the general-purpose role used by anyone on EASI's shared infrastructure, e.g. JupyterHub notebook sessions.
- `csa-parkes-team-service-argo` — a role for a specific Argo/Kubernetes-based service, set up for astronomy users to run their own applications on EASI's Kubernetes cluster.

To grant both of these read/write access, use a bucket policy like:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AddAccountReadWrite",
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                    "arn:aws:iam::444488357543:role/easihub-csiro-csiro-easihub-client",
                    "arn:aws:iam::444488357543:role/csa-parkes-team-service-argo"
                ]
            },
            "Action": [
                "s3:PutObject",
                "s3:GetObjectAcl",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:PutObjectAcl",
                "s3:AbortMultipartUpload",
                "s3:ListMultipartUploadParts",
                "s3:PutObjectTagging",
                "s3:GetObjectTagging"
            ],
            "Resource": "arn:aws:s3:::emu-data-bucket-2026/*"
        },
        {
            "Sid": "AddAccountReadC",
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                    "arn:aws:iam::444488357543:role/easihub-csiro-csiro-easihub-client",
                    "arn:aws:iam::444488357543:role/csa-parkes-team-service-argo"
                ]
            },
            "Action": [
                "s3:GetBucketLocation",
                "s3:ListBucket",
                "s3:ListBucketMultipartUploads"
            ],
            "Resource": "arn:aws:s3:::emu-data-bucket-2026"
        }
    ]
}
```

**Reading this policy:**

- `"Principal": {"AWS": [...]}` lists *who* the permissions apply to. Both entries are IAM roles belonging to CSIRO's EASI AWS account (account ID `444488357543`), not the bucket owner's own account — one for general EASI/JupyterHub access, one for the Parkes-team Kubernetes/Argo service.
- The **first statement** applies to `.../emu-data-bucket-2026/*` — note the trailing `/*`, meaning "every object inside the bucket." Its actions (`PutObject`, `GetObject`, `DeleteObject`, etc.) are all **object-level** operations: reading, writing, deleting, and tagging individual files.
- The **second statement** applies to the bucket itself (no `/*`), and its actions (`ListBucket`, `GetBucketLocation`, etc.) are **bucket-level** operations: listing what's inside it and finding out which region it lives in. S3 splits permissions this way, so both statements are needed for these roles to both browse and read/write the data.
- Replace `emu-data-bucket-2026` with the actual bucket name, and confirm the exact role ARNs (Amazon Resource Names — unique identifier strings for AWS resources and identities) against current EASI documentation or with whoever administers the relevant service, since account IDs and role names can change over time or vary by team.
- To add further Kubernetes-based services or teams later, add their role ARNs to both `"AWS": [...]` lists rather than creating new statements.

---

## 9. Access the bucket from EASI

Once the bucket policy above is in place, EASI can already reach the bucket — but exactly what's needed depends on *where* the access is happening from.

### From a remote system (a laptop or an HPC cluster) accessing an EASI account

If working from outside EASI — e.g. a personal laptop or an HPC login node like Virga — that machine has no built-in identity that EASI recognises, so it needs to borrow one via the EASI credentials page:

1. Go to https://www.csiro.easi-eo.solutions/credentials/ and export the temporary credentials shown there into the terminal. These credentials refresh regularly (they may change daily), so this step needs repeating periodically — a running EASI Jupyter session is required for the credentials page to issue a valid set.
2. Optionally, save them into a file such as `~/.bash_easi_cloud_keys` so they can be reloaded with `source ~/.bash_easi_cloud_keys` instead of re-pasting them each time.
3. Once the credentials are exported into that remote shell, access the bucket exactly as anywhere else:

```bash
aws s3 ls s3://emu-data-bucket-2026/
```

### From inside an EASI JupyterHub session

No manual credential step is needed here. A notebook running inside EASI's JupyterHub is already operating under the `easihub-csiro-csiro-easihub-client` role — the same role already granted access in the bucket policy above — so the identical command works immediately, with nothing to export or configure first:

```bash
aws s3 ls s3://emu-data-bucket-2026/
```

If this lists the uploaded files (from either environment), the whole chain — AWS account and IAM user → CLI install → login → bucket creation → upload → bucket policy → EASI access — is working end to end, and the data can be read directly into analysis code (`xarray`, `astropy`, `dask`, etc.), without needing to be downloaded to local disk first.

For apps running on Kubernetes under the Argo service role (`csa-parkes-team-service-argo`), the equivalent credentials are typically injected automatically by the cluster rather than exported manually — check with whoever set up that service for the specifics.

---

## Quick troubleshooting checklist

- **`aws: command not found`** → the `PATH` update from Step 2 didn't take effect in this shell; re-run `export PATH="$HOME/.local/bin:$PATH"` or open a new terminal.
- **`aws sts get-caller-identity` fails / expired token** → the `aws login` session has timed out; log in again.
- **`AccessDenied` when EASI tries to read the bucket** → check the bucket policy's `Resource` fields match the actual bucket name exactly, and that both the object-level and bucket-level statements are present, with the correct role ARN(s) listed.
- **Bucket name rejected on creation** → S3 bucket names must be globally unique, lowercase, 3–63 characters, and contain no underscores, spaces, or uppercase letters.
- **Costs higher than expected** → run `aws s3 ls s3://<bucket>/ --recursive --summarize` to check total object count/size, and confirm which storage class objects are actually in (S3 Standard is the default and the most expensive per GB).
