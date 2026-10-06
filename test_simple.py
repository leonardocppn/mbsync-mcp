#!/usr/bin/env python3
"""Smoke test for the mbsync-mcp server.

Runs against the accounts declared in ~/.mbsyncrc on this machine. It parses
the configuration, resolves each account by name and by address, and counts
the emails in the Maildir folders of each account, without writing anything
or opening IMAP connections.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import (
    parse_mbsyncrc,
    get_accounts,
    resolve_account,
    get_account_maildir_path,
    get_maildir_folders,
    count_emails_maildir,
    get_folder_path,
)


def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_config():
    """Parse ~/.mbsyncrc."""
    header("Parsing ~/.mbsyncrc")

    accounts = parse_mbsyncrc().get("accounts", [])
    print(f"\n{len(accounts)} account(s) found:")
    for acc in accounts:
        print(f"\n  Channel: {acc.get('name', 'unknown')}")
        print(f"  Email:   {acc.get('email', 'N/A')}")
        print(f"  Host:    {acc.get('host', 'N/A')}")
        print(f"  Path:    {acc.get('local_path', 'N/A')}")

    return len(accounts) > 0


def test_resolve():
    """Resolve each account by channel name and by address, ignoring case, and check that partial names are rejected."""
    header("Account resolution")

    accounts = get_accounts()
    if not accounts:
        print("\n  No account configured")
        return False

    ok = True
    for acc in accounts:
        name = acc.get("name", "")
        email_addr = acc.get("email", "")
        queries = [q for q in (name, email_addr, name.upper()) if q]
        for query in queries:
            resolved = resolve_account(query)
            found = resolved.get("email", "N/A") if resolved else "not found"
            print(f"\n  '{query}' -> {found}")
            if not resolved or resolved.get("email") != email_addr:
                ok = False

    # A piece of a name or address could belong to another account, so it must not resolve
    for query in ("nonexistent-account-xyz", accounts[0].get("name", "")[:-1], "@", ""):
        if resolve_account(query) is not None:
            print(f"\n  '{query}' resolved, but should not have")
            ok = False

    return ok


def test_folders():
    """List folders and count messages."""
    header("Folders")

    accounts = get_accounts()
    if not accounts:
        print("\n  No account configured")
        return False

    for acc in accounts:
        path = get_account_maildir_path(acc.get("name", ""))
        if not path:
            print(f"\n  {acc.get('email', 'unknown')}: Maildir not found")
            continue
        print(f"\n  {acc.get('email', 'unknown')}")
        for folder in get_maildir_folders(path):
            total, unread = count_emails_maildir(folder["path"])
            print(f"    • {folder['name']}: {total} email ({unread} unread)")

    return True


def test_inbox():
    """Count the emails in the INBOX of each account."""
    header("INBOX")

    accounts = get_accounts()
    if not accounts:
        print("\n  No account configured")
        return False

    for acc in accounts:
        folder_path = get_folder_path(acc.get("name", ""), "INBOX")
        if not folder_path:
            print(f"\n  {acc.get('email', 'unknown')}: INBOX not found")
            continue
        total, unread = count_emails_maildir(folder_path)
        print(f"\n  {acc.get('email', 'unknown')}")
        print(f"  INBOX: {total} total, {unread} unread")

    return True


if __name__ == "__main__":
    print("\n🧪 mbsync-mcp smoke test")

    results = [
        ("Config", test_config()),
        ("Resolve", test_resolve()),
        ("Folders", test_folders()),
        ("INBOX", test_inbox()),
    ]

    header("Results")
    for name, passed in results:
        print(f"  {'✅' if passed else '❌'} {name}")

    all_passed = all(passed for _, passed in results)
    print(f"\n{'All tests passed' if all_passed else 'Some tests failed'}\n")
    sys.exit(0 if all_passed else 1)
