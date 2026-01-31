#!/usr/bin/env python3
"""
mbsync MCP Server - MCP server for email access via mbsync/Maildir

Reads emails from Maildir folders synchronized with mbsync.
Configuration: ~/.mbsyncrc
"""

import asyncio
import email
import imaplib
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from email.policy import default as email_policy
from email.utils import parsedate_to_datetime
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

# =============================================================================
# Configuration
# =============================================================================

server = Server("mbsync-mcp")
SERVICE_NAME = "mbsync-mcp"
MAIL_BASE_PATH = Path.home() / "Mail"
DEBUG = os.environ.get("MBSYNC_DEBUG", "").lower() in ("1", "true")

_mbsync_config_cache = None


def debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}", flush=True)


# =============================================================================
# Parsing ~/.mbsyncrc
# =============================================================================

def parse_mbsyncrc() -> dict:
    """Parse ~/.mbsyncrc and return account configuration."""
    global _mbsync_config_cache
    if _mbsync_config_cache is not None:
        return _mbsync_config_cache

    mbsyncrc_path = Path.home() / ".mbsyncrc"
    if not mbsyncrc_path.exists():
        return {"accounts": []}

    content = mbsyncrc_path.read_text()
    accounts = []
    current = {}

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        if line.startswith('IMAPAccount '):
            if current:
                accounts.append(current)
            current = {'name': line.split()[1]}
        elif current:
            if line.startswith('Host '):
                current['host'] = line.split()[1]
            elif line.startswith('User '):
                current['email'] = line.split()[1]
            elif line.startswith('PassCmd '):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    current['pass_cmd'] = match.group(1)

    if current:
        accounts.append(current)

    # Associate Maildir paths
    for acc in accounts:
        name = acc['name']
        for pattern, key in [
            (rf'MaildirStore\s+{name}-local.*?Path\s+([^\n]+)', 'local_path'),
            (rf'MaildirStore\s+{name}-local.*?Inbox\s+([^\n]+)', 'inbox')
        ]:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                path = match.group(1).strip()
                if path.startswith('~/'):
                    path = str(Path.home() / path[2:])
                acc[key] = path

    _mbsync_config_cache = {"accounts": accounts}
    return _mbsync_config_cache


def get_accounts() -> list[dict]:
    return parse_mbsyncrc().get("accounts", [])


def resolve_account(account_input: str) -> dict | None:
    """Resolve an account from partial input (email, name, or substring)."""
    lower = account_input.lower()
    for acc in get_accounts():
        for field in ('email', 'name'):
            val = acc.get(field, '').lower()
            if val == lower or lower in val:
                return acc
    return None


def get_account_maildir_path(account_input: str) -> Path | None:
    acc = resolve_account(account_input)
    if acc and 'local_path' in acc:
        path = Path(acc['local_path'])
        return path if path.exists() else None
    return None


# =============================================================================
# Maildir Functions
# =============================================================================

def is_email_read(filepath: Path) -> bool:
    """Check S (Seen) flag in Maildir filename."""
    if filepath.parent.name == "new":
        return False
    if ":2," in filepath.name:
        return "S" in filepath.name.split(":2,")[1]
    return False


def get_maildir_folders(base_path: Path) -> list[dict]:
    """List Maildir folders."""
    folders = []
    inbox = base_path / "INBOX"
    if inbox.exists():
        folders.append({"name": "INBOX", "path": inbox})

    for item in base_path.iterdir():
        if not item.is_dir() or item.name == "INBOX":
            continue
        if (item / "cur").exists() or (item / "new").exists():
            folders.append({"name": item.name, "path": item})
        else:
            for sub in item.iterdir():
                if sub.is_dir() and ((sub / "cur").exists() or (sub / "new").exists()):
                    folders.append({"name": f"{item.name}/{sub.name}", "path": sub})
    return folders


def count_emails_maildir(folder_path: Path) -> tuple[int, int]:
    """Count total and unread emails."""
    total = unread = 0
    new_path = folder_path / "new"
    if new_path.exists():
        count = sum(1 for f in new_path.iterdir() if f.is_file())
        total += count
        unread += count

    cur_path = folder_path / "cur"
    if cur_path.exists():
        for f in cur_path.iterdir():
            if f.is_file():
                total += 1
                if not is_email_read(f):
                    unread += 1
    return total, unread


def parse_maildir_email(filepath: Path, index: int = 0, full_body: bool = False) -> dict:
    """Parse a Maildir email file."""
    try:
        with open(filepath, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=email_policy)

        body = ""
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition", ""))

                if "attachment" in disp:
                    attachments.append({
                        "filename": part.get_filename() or "unnamed",
                        "type": ctype
                    })
                elif ctype in ("text/plain", "text/html") and not body:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body = payload.decode(charset, errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body = payload.decode(charset, errors="replace") if isinstance(payload, bytes) else str(payload)

        result = {
            "index": index,
            "filepath": str(filepath),
            "filename": filepath.name,
            "from": str(msg.get("From", "")),
            "to": str(msg.get("To", "")),
            "cc": str(msg.get("Cc", "")),
            "subject": str(msg.get("Subject", "")),
            "date": str(msg.get("Date", "")),
            "message_id": str(msg.get("Message-ID", "")),
            "body_preview": body[:200] if body else "",
            "attachments": attachments,
            "is_read": is_email_read(filepath)
        }
        if full_body:
            result["body"] = body
        return result

    except Exception as e:
        return {"index": index, "filepath": str(filepath), "error": str(e)}


def get_maildir_emails(folder_path: Path, limit: int = 50, unread_only: bool = False,
                       date_from: datetime = None, date_to: datetime = None) -> list[dict]:
    """Read emails from Maildir folder with filters."""
    all_files = []
    new_path = folder_path / "new"
    if new_path.exists():
        all_files.extend(f for f in new_path.iterdir() if f.is_file())

    if not unread_only:
        cur_path = folder_path / "cur"
        if cur_path.exists():
            all_files.extend(f for f in cur_path.iterdir() if f.is_file())

    all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    emails = []
    for i, filepath in enumerate(all_files):
        data = parse_maildir_email(filepath, i)
        if "error" in data:
            continue
        if unread_only and data.get("is_read"):
            continue

        if date_from or date_to:
            try:
                edate = parsedate_to_datetime(data.get("date", "")).replace(tzinfo=None)
                if date_from and edate < date_from:
                    continue
                if date_to and edate > date_to:
                    continue
            except:
                pass

        emails.append(data)
        if len(emails) >= limit:
            break
    return emails


def search_in_maildir(folder_path: Path, query: str, field: str = "all", limit: int = 20) -> list[dict]:
    """Search emails by query."""
    q = query.lower()
    results = []
    for email_data in get_maildir_emails(folder_path, limit=9999):
        match = False
        if field in ("from", "all") and q in email_data.get("from", "").lower():
            match = True
        if field in ("subject", "all") and q in email_data.get("subject", "").lower():
            match = True
        if field in ("body", "all") and q in email_data.get("body_preview", "").lower():
            match = True
        if match:
            results.append(email_data)
            if len(results) >= limit:
                break
    return results


def get_folder_path(account_input: str, folder_name: str) -> Path | None:
    """Find path of a Maildir folder."""
    base = get_account_maildir_path(account_input)
    if not base:
        return None

    folder_path = base / folder_name
    if folder_path.exists():
        return folder_path

    if "/" in folder_name:
        folder_path = base
        for part in folder_name.split("/"):
            folder_path = folder_path / part
        if folder_path.exists():
            return folder_path
    return None


def get_email_by_index(folder_path: Path, index: int) -> dict | None:
    """Get email by index with full body."""
    emails = get_maildir_emails(folder_path, limit=index + 1)
    if index < len(emails):
        return parse_maildir_email(Path(emails[index]["filepath"]), index, full_body=True)
    return None


def parse_date_filter(date_str: str) -> datetime | None:
    if not date_str:
        return None
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    presets = {
        "today": today,
        "yesterday": today - timedelta(days=1),
        "last-7-days": today - timedelta(days=7),
        "last-30-days": today - timedelta(days=30)
    }
    if date_str in presets:
        return presets[date_str]
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None


# =============================================================================
# IMAP Helper
# =============================================================================

def get_imap_credentials(account: dict) -> tuple[str, str] | None:
    """Get IMAP credentials from keyring or PassCmd."""
    email_addr = account.get('email', '')

    try:
        import keyring
        pwd = keyring.get_password(SERVICE_NAME, email_addr)
        if pwd:
            return email_addr, pwd
    except:
        pass

    pass_cmd = account.get('pass_cmd', '')
    if pass_cmd:
        try:
            result = subprocess.run(pass_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return email_addr, result.stdout.strip()
        except:
            pass
    return None


def connect_imap(account: dict) -> imaplib.IMAP4_SSL | None:
    creds = get_imap_credentials(account)
    if not creds:
        return None
    try:
        mail = imaplib.IMAP4_SSL(account.get('host', 'imap.gmail.com'))
        mail.login(*creds)
        return mail
    except Exception as e:
        debug(f"IMAP error: {e}")
        return None


def find_special_folders(mail: imaplib.IMAP4_SSL, host: str) -> dict:
    """Find special folders (Trash, All Mail) for provider."""
    is_gmail = 'gmail' in host.lower()
    result = {
        'trash': '[Gmail]/Cestino' if is_gmail else 'Trash',
        'all_mail': '[Gmail]/Tutti i messaggi' if is_gmail else 'Archive'
    }

    try:
        status, folders = mail.list()
        if status == 'OK':
            for f in folders:
                decoded = f.decode() if isinstance(f, bytes) else str(f)
                match = re.search(r'"([^"]+)"$', decoded)
                if match:
                    name = match.group(1)
                    if '\\Trash' in decoded:
                        result['trash'] = name
                    elif '\\All' in decoded:
                        result['all_mail'] = name
    except:
        pass
    return result


def get_sorted_uids(mail: imaplib.IMAP4_SSL, folder: str) -> list[str]:
    """Get UIDs sorted by date (most recent first)."""
    mail.select(folder)
    try:
        status, data = mail.uid('SORT', '(REVERSE DATE)', 'UTF-8', 'ALL')
        if status == 'OK' and data[0]:
            return [u.decode() if isinstance(u, bytes) else u for u in data[0].split()]
    except:
        pass

    status, data = mail.uid('SEARCH', None, 'ALL')
    if status == 'OK' and data[0]:
        uids = data[0].split()
        return [u.decode() if isinstance(u, bytes) else u for u in reversed(uids)]
    return []


def get_uid_by_index(mail: imaplib.IMAP4_SSL, folder: str, index: int) -> tuple[str, str] | None:
    """Get UID and subject by index (0 = most recent)."""
    uids = get_sorted_uids(mail, folder)
    if index >= len(uids):
        return None

    uid = uids[index]
    subject = ""
    try:
        _, data = mail.uid('FETCH', uid, '(BODY.PEEK[HEADER.FIELDS (SUBJECT)])')
        if data and data[0] and len(data[0]) > 1:
            raw = data[0][1]
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            subject = raw.replace('\r\n', ' ').replace('\n', ' ')[:60]
    except:
        pass
    return uid, subject


# =============================================================================
# IMAP Operations
# =============================================================================

def sync_account(account_name: str, timeout: int = 300) -> dict:
    """Run mbsync to synchronize."""
    try:
        result = subprocess.run(["mbsync", account_name], capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return {"success": True, "message": f"Sync {account_name} completed"}
        return {"success": False, "error": result.stderr or "Error"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s"}
    except FileNotFoundError:
        return {"success": False, "error": "mbsync not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def imap_move(mail: imaplib.IMAP4_SSL, folder: str, uid: str, dest: str) -> bool:
    """Move email via IMAP MOVE."""
    mail.select(folder)
    status, _ = mail.uid('MOVE', uid, f'"{dest}"')
    return status == 'OK'


def archive_email_by_index(account: dict, folder: str, index: int) -> dict:
    mail = connect_imap(account)
    if not mail:
        return {"error": "IMAP connection failed"}

    try:
        result = get_uid_by_index(mail, folder, index)
        if not result:
            return {"error": f"Email not found at index {index}"}

        uid, subject = result
        special = find_special_folders(mail, account.get('host', ''))

        if imap_move(mail, folder, uid, special['all_mail']):
            return {"success": True, "message": f"Archived: {subject[:50]}"}
        return {"error": "MOVE failed"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            mail.logout()
        except:
            pass


def delete_email_by_index(account: dict, folder: str, index: int) -> dict:
    mail = connect_imap(account)
    if not mail:
        return {"error": "IMAP connection failed"}

    try:
        result = get_uid_by_index(mail, folder, index)
        if not result:
            return {"error": f"Email not found at index {index}"}

        uid, subject = result
        special = find_special_folders(mail, account.get('host', ''))

        if imap_move(mail, folder, uid, special['trash']):
            return {"success": True, "message": f"Deleted: {subject[:50]}"}
        return {"error": "MOVE failed"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            mail.logout()
        except:
            pass


def move_email_by_index(account: dict, source: str, dest: str, index: int) -> dict:
    mail = connect_imap(account)
    if not mail:
        return {"error": "IMAP connection failed"}

    try:
        result = get_uid_by_index(mail, source, index)
        if not result:
            return {"error": f"Email not found at index {index}"}

        uid, subject = result
        if imap_move(mail, source, uid, dest):
            return {"success": True, "message": f"Moved to {dest}: {subject[:50]}"}
        return {"error": "MOVE failed"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            mail.logout()
        except:
            pass


def cleanup_batch(account: dict, folder: str, delete_idxs: list, archive_idxs: list) -> dict:
    mail = connect_imap(account)
    if not mail:
        return {"error": "IMAP connection failed"}

    try:
        all_idxs = sorted(set(delete_idxs + archive_idxs))
        uids = get_sorted_uids(mail, folder)
        special = find_special_folders(mail, account.get('host', ''))

        uid_map = {i: uids[i] for i in all_idxs if i < len(uids)}
        deleted = archived = 0
        errors = []

        for idx in delete_idxs:
            if idx in uid_map:
                if imap_move(mail, folder, uid_map[idx], special['trash']):
                    deleted += 1
                else:
                    errors.append(f"Delete [{idx}] failed")

        for idx in archive_idxs:
            if idx in uid_map:
                if imap_move(mail, folder, uid_map[idx], special['all_mail']):
                    archived += 1
                else:
                    errors.append(f"Archive [{idx}] failed")

        return {"deleted": deleted, "archived": archived, "errors": errors}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            mail.logout()
        except:
            pass


# =============================================================================
# MCP Resources
# =============================================================================

@server.list_resources()
async def list_resources() -> list[Resource]:
    resources = [Resource(
        uri="mbsync://accounts",
        name="Configured accounts",
        description="Account list from ~/.mbsyncrc",
        mimeType="application/json"
    )]
    for acc in get_accounts():
        resources.append(Resource(
            uri=f"mbsync://account/{acc.get('name')}/folders",
            name=f"Folders {acc.get('email', '')}",
            description=f"Folders for {acc.get('email', '')}",
            mimeType="application/json"
        ))
    return resources


@server.read_resource()
async def read_resource(uri: str) -> str:
    uri = str(uri)
    if uri == "mbsync://accounts":
        return json.dumps(get_accounts(), indent=2, ensure_ascii=False)

    if uri.startswith("mbsync://account/") and uri.endswith("/folders"):
        name = uri.replace("mbsync://account/", "").replace("/folders", "")
        path = get_account_maildir_path(name)
        if path:
            folders = get_maildir_folders(path)
            return json.dumps([{"name": f["name"]} for f in folders], indent=2)
        return json.dumps({"error": "Account not found"})

    return json.dumps({"error": "Resource not found"})


# =============================================================================
# MCP Tools - Definitions
# =============================================================================

TOOL_DEFS = [
    ("list_accounts", "List email accounts configured in ~/.mbsyncrc.", {}),
    ("list_folders", "List available folders of an email account.", {
        "account": ("string", "Email or account name (e.g., myaccount, user@example.com)", True)
    }),
    ("count_emails", "Count the number of emails in a folder.", {
        "account": ("string", "Email or account name", True),
        "folder": ("string", "Folder name (e.g., INBOX)", True)
    }),
    ("get_unread_emails", "Return unread emails in a folder.", {
        "account": ("string", "Email or account name", True),
        "folder": ("string", "Folder name (e.g., INBOX)", True),
        "limit": ("integer", "Maximum number of results (default: 50)", False)
    }),
    ("get_emails", "Read emails with advanced date and status filters.", {
        "account": ("string", "Email or account name", True),
        "folder": ("string", "Folder name (default: INBOX)", False),
        "date_from": ("string", "Start date: today, yesterday, last-7-days, last-30-days, or YYYY-MM-DD", False),
        "date_to": ("string", "End date (optional)", False),
        "unread_only": ("boolean", "If true, only unread emails (default: false)", False),
        "limit": ("integer", "Maximum number of results (default: 50)", False),
        "auto_sync": ("boolean", "If true, sync before reading (default: false)", False)
    }),
    ("get_email_details", "Read complete details of a specific email.", {
        "account": ("string", "Email or account name", True),
        "folder": ("string", "Folder name", True),
        "index": ("integer", "Email index (from 0)", True)
    }),
    ("search_emails", "Search emails by sender, subject, or content.", {
        "account": ("string", "Email or account name", True),
        "folder": ("string", "Folder name", True),
        "query": ("string", "Text to search", True),
        "field": ("string", "Where to search (default: all)", False, ["from", "subject", "body", "all"]),
        "limit": ("integer", "Maximum number of results (default: 20)", False)
    }),
    ("get_inbox_summary", "Inbox summary: counts, unread, top senders.", {
        "account": ("string", "Email or account name", True)
    }),
    ("daily_briefing", "Complete daily briefing: counts, unread emails, top senders.", {
        "account": ("string", "Email or account name", True),
        "date": ("string", "Date: today, yesterday, last-7-days, or YYYY-MM-DD (default: today)", False),
        "show_details": ("boolean", "Show detailed list (default: true)", False),
        "limit": ("integer", "Maximum number of emails (default: 20)", False),
        "auto_sync": ("boolean", "If true, sync before reading (default: true)", False)
    }),
    ("sync_account", "Sync an account via mbsync. Downloads new emails from IMAP server.", {
        "account": ("string", "mbsync account/channel name (e.g., myaccount)", True),
        "timeout": ("integer", "Timeout in seconds (default: 300)", False)
    }),
    ("move_email", "Move an email to another folder via IMAP.", {
        "account": ("string", "Email or account name", True),
        "source_folder": ("string", "Source folder (e.g., INBOX)", True),
        "dest_folder": ("string", "Destination folder", True),
        "index": ("integer", "Email index", True)
    }),
    ("delete_email", "Move an email to trash via IMAP.", {
        "account": ("string", "Email or account name", True),
        "folder": ("string", "Current folder (e.g., INBOX)", True),
        "index": ("integer", "Email index", True)
    }),
    ("archive_email", "Archive an email via IMAP. On Gmail removes from INBOX.", {
        "account": ("string", "Email or account name", True),
        "folder": ("string", "Current folder (e.g., INBOX)", True),
        "index": ("integer", "Email index", True)
    }),
    ("cleanup_batch", "Batch operations: delete and/or archive multiple emails.", {
        "account": ("string", "Email or account name", True),
        "folder": ("string", "Folder (default: INBOX)", False),
        "delete_indexes": ("array", "Email indexes to delete", False),
        "archive_indexes": ("array", "Email indexes to archive", False)
    }),
]


def build_tools() -> list[Tool]:
    tools = []
    for name, desc, params in TOOL_DEFS:
        props = {}
        required = []
        for pname, pdef in params.items():
            ptype, pdesc, preq = pdef[:3]
            prop = {"type": ptype, "description": pdesc}
            if len(pdef) > 3:
                prop["enum"] = pdef[3]
            if ptype == "array":
                prop["items"] = {"type": "integer"}
            props[pname] = prop
            if preq:
                required.append(pname)
        tools.append(Tool(
            name=name,
            description=desc,
            inputSchema={"type": "object", "properties": props, "required": required}
        ))
    return tools


@server.list_tools()
async def list_tools() -> list[Tool]:
    return build_tools()


# =============================================================================
# MCP Tools - Handler
# =============================================================================

def format_email_list(emails: list, title: str) -> str:
    if not emails:
        return f"No emails found"
    lines = [f"{title} ({len(emails)}):\n"]
    for e in emails:
        status = "✓" if e.get('is_read') else "✉"
        lines.append(f"[{e['index']}] [{status}] {e['subject'][:60]}")
        lines.append(f"    From: {e['from'][:50]}")
        lines.append(f"    {e['date'][:30]}\n")
    return "\n".join(lines)


def get_top_senders(emails: list, n: int = 5) -> list[tuple]:
    senders = Counter()
    for e in emails:
        f = e.get('from', '')
        name = f.split('<')[0].strip().strip('"') if '<' in f else f
        senders[name] += 1
    return senders.most_common(n)


TOOL_HANDLERS = {}


def tool(name):
    def decorator(fn):
        TOOL_HANDLERS[name] = fn
        return fn
    return decorator


@tool("list_accounts")
def handle_list_accounts(args):
    accounts = get_accounts()
    if not accounts:
        return "No accounts configured in ~/.mbsyncrc"
    lines = ["Configured accounts:\n"]
    for acc in accounts:
        lines.append(f"• {acc.get('name', 'unknown')}")
        lines.append(f"  Email: {acc.get('email', 'N/A')}")
        lines.append(f"  Host: {acc.get('host', 'N/A')}")
        lines.append(f"  Path: {acc.get('local_path', 'N/A')}\n")
    return "\n".join(lines)


@tool("list_folders")
def handle_list_folders(args):
    path = get_account_maildir_path(args.get("account", ""))
    if not path:
        return f"Account not found: {args.get('account')}"
    folders = get_maildir_folders(path)
    lines = [f"Folders for {args.get('account')}:\n"]
    for f in folders:
        total, unread = count_emails_maildir(f["path"])
        lines.append(f"• {f['name']}: {total} emails ({unread} unread)")
    return "\n".join(lines)


@tool("count_emails")
def handle_count_emails(args):
    path = get_folder_path(args.get("account", ""), args.get("folder", "INBOX"))
    if not path:
        return f"Folder not found"
    total, unread = count_emails_maildir(path)
    return f"{args.get('folder')}: {total} total, {unread} unread"


@tool("get_unread_emails")
def handle_get_unread_emails(args):
    path = get_folder_path(args.get("account", ""), args.get("folder", "INBOX"))
    if not path:
        return f"Folder not found"
    emails = get_maildir_emails(path, args.get("limit", 50), unread_only=True)
    return format_email_list(emails, f"Unread emails in {args.get('folder')}")


@tool("get_emails")
def handle_get_emails(args):
    account = args.get("account", "")
    folder = args.get("folder", "INBOX")

    if args.get("auto_sync"):
        acc = resolve_account(account)
        if acc:
            sync_account(acc.get('name', account))

    path = get_folder_path(account, folder)
    if not path:
        return f"Folder not found"

    emails = get_maildir_emails(
        path,
        args.get("limit", 50),
        args.get("unread_only", False),
        parse_date_filter(args.get("date_from", "")),
        parse_date_filter(args.get("date_to", ""))
    )
    return format_email_list(emails, f"Emails in {folder}")


@tool("get_email_details")
def handle_get_email_details(args):
    path = get_folder_path(args.get("account", ""), args.get("folder", "INBOX"))
    if not path:
        return "Folder not found"

    email_data = get_email_by_index(path, args.get("index", 0))
    if not email_data:
        return f"Email not found at index {args.get('index')}"

    status = "Read" if email_data.get('is_read') else "Unread"
    lines = [
        f"Email #{args.get('index')} [{status}]\n",
        f"Message-ID: {email_data.get('message_id', 'N/A')}",
        f"From: {email_data.get('from', 'N/A')}",
        f"To: {email_data.get('to', 'N/A')}",
    ]
    if email_data.get('cc'):
        lines.append(f"Cc: {email_data.get('cc')}")
    lines.extend([
        f"Subject: {email_data.get('subject', 'N/A')}",
        f"Date: {email_data.get('date', 'N/A')}"
    ])

    if email_data.get('attachments'):
        lines.append(f"\nAttachments: {len(email_data['attachments'])}")
        for att in email_data['attachments']:
            lines.append(f"  • {att['filename']} ({att['type']})")

    lines.append(f"\n--- Body ---\n{email_data.get('body', '')[:5000]}")
    return "\n".join(lines)


@tool("search_emails")
def handle_search_emails(args):
    path = get_folder_path(args.get("account", ""), args.get("folder", "INBOX"))
    if not path:
        return "Folder not found"

    results = search_in_maildir(path, args.get("query", ""), args.get("field", "all"), args.get("limit", 20))
    return format_email_list(results, f"Results for '{args.get('query')}'")


@tool("get_inbox_summary")
def handle_get_inbox_summary(args):
    path = get_folder_path(args.get("account", ""), "INBOX")
    if not path:
        return "Account not found"

    total, unread = count_emails_maildir(path)
    emails = get_maildir_emails(path, limit=100)
    top = get_top_senders(emails)

    acc = resolve_account(args.get("account", ""))
    email_addr = acc.get('email', args.get('account')) if acc else args.get('account')

    lines = [
        f"INBOX Summary - {email_addr}\n",
        f"Total: {total} emails",
        f"Unread: {unread}\n",
        "Top senders:"
    ]
    for sender, count in top:
        lines.append(f"   • {sender}: {count}")
    return "\n".join(lines)


@tool("daily_briefing")
def handle_daily_briefing(args):
    account = args.get("account", "")
    acc = resolve_account(account)

    if args.get("auto_sync", True) and acc:
        sync_account(acc.get('name', account))

    path = get_folder_path(account, "INBOX")
    if not path:
        return "Account not found"

    email_addr = acc.get('email', account) if acc else account
    total, unread = count_emails_maildir(path)
    emails = get_maildir_emails(path, limit=100)
    unread_emails = [e for e in emails if not e.get('is_read', False)]
    top = get_top_senders(emails)
    limit = args.get("limit", 20)

    lines = [
        f"Briefing - {email_addr}",
        "=" * 60,
        f"\nINBOX: {total} total emails, {unread} unread\n",
        "Top senders:"
    ]
    for sender, count in top:
        lines.append(f"   • {sender} ({count})")
    lines.append("")

    if args.get("show_details", True) and unread_emails:
        lines.append(f"Emails to read ({min(limit, len(unread_emails))} of {len(unread_emails)}):\n")
        for i, e in enumerate(unread_emails[:limit], 1):
            lines.append(f"{i}. [{e['index']}] {e['subject'][:70]}")
            lines.append(f"   From: {e['from'][:60]}")
            lines.append(f"   Date: {e['date'][:40]}")
            if e.get('body_preview'):
                lines.append(f"   {e['body_preview'][:80]}...")
            lines.append("")

        if len(unread_emails) > limit:
            lines.append(f"... and {len(unread_emails) - limit} more unread emails")

    return "\n".join(lines)


@tool("sync_account")
def handle_sync_account(args):
    acc = resolve_account(args.get("account", ""))
    if not acc:
        return f"Account not found: {args.get('account')}"

    result = sync_account(acc.get('name', args.get('account')), args.get("timeout", 300))
    if result.get('success'):
        return f"✅ {result['message']}"
    return f"❌ {result.get('error')}"


@tool("move_email")
def handle_move_email(args):
    acc = resolve_account(args.get("account", ""))
    if not acc:
        return "Account not found"

    result = move_email_by_index(acc, args.get("source_folder", "INBOX"), args.get("dest_folder", ""), args.get("index", 0))
    return f"✅ {result['message']}" if result.get('success') else f"❌ {result.get('error')}"


@tool("delete_email")
def handle_delete_email(args):
    acc = resolve_account(args.get("account", ""))
    if not acc:
        return "Account not found"

    result = delete_email_by_index(acc, args.get("folder", "INBOX"), args.get("index", 0))
    return "🗑️ Email moved to trash" if result.get('success') else f"❌ {result.get('error')}"


@tool("archive_email")
def handle_archive_email(args):
    acc = resolve_account(args.get("account", ""))
    if not acc:
        return "Account not found"

    result = archive_email_by_index(acc, args.get("folder", "INBOX"), args.get("index", 0))
    return "📦 Email archived" if result.get('success') else f"❌ {result.get('error')}"


@tool("cleanup_batch")
def handle_cleanup_batch(args):
    acc = resolve_account(args.get("account", ""))
    if not acc:
        return "Account not found"

    result = cleanup_batch(
        acc,
        args.get("folder", "INBOX"),
        args.get("delete_indexes", []),
        args.get("archive_indexes", [])
    )

    if "error" in result:
        return f"❌ {result['error']}"

    lines = [
        "Operations completed:",
        f"🗑️ Deleted: {result.get('deleted', 0)}",
        f"📦 Archived: {result.get('archived', 0)}"
    ]
    if result.get('errors'):
        lines.append(f"\n⚠️ Errors: {len(result['errors'])}")
        for err in result['errors'][:5]:
            lines.append(f"  • {err}")
    return "\n".join(lines)


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    handler = TOOL_HANDLERS.get(name)
    if handler:
        result = handler(arguments)
        return [TextContent(type="text", text=result)]
    return [TextContent(type="text", text=f"Unrecognized tool: {name}")]


# =============================================================================
# Main
# =============================================================================

async def main():
    accounts = get_accounts()
    print(f"mbsync MCP Server - {len(accounts)} accounts configured", flush=True)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
