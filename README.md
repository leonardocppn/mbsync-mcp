# mbsync-mcp

A [Model Context Protocol](https://modelcontextprotocol.io/) server that gives an AI assistant access to the mail [mbsync](https://isync.sourceforge.io/) already keeps on your disk.

Reading happens entirely on the local Maildir, so it works offline and sends nothing over the network. Moving, archiving and deleting go out over IMAP, because that is where the mailbox actually lives; after a write succeeds the server runs mbsync for that account, so the local copy catches up on its own.

Accounts, hosts and Maildir paths are read from `~/.mbsyncrc`. The server keeps no configuration of its own, and reading your mail requires no password.

## Requirements

Linux or macOS (on Windows, only under WSL), Python 3.10 or later, mbsync installed and configured, `mcp>=1.0.0`, plus `keyring` if you want the write tools.

## Installation

```bash
git clone https://github.com/leonardocppn/mbsync-mcp.git
cd mbsync-mcp
pip install -r requirements.txt
```

## Configuration

The server parses `~/.mbsyncrc` to discover your accounts, so whatever mbsync syncs is already visible to it. A minimal account looks like this:

```
IMAPAccount myaccount
Host imap.gmail.com
User user@gmail.com
PassCmd "pass email/gmail"
SSLType IMAPS

IMAPStore myaccount-remote
Account myaccount

MaildirStore myaccount-local
Path ~/Mail/myaccount/
Inbox ~/Mail/myaccount/INBOX

Channel myaccount
Far :myaccount-remote:
Near :myaccount-local:
Patterns *
Create Both
SyncState *
```

Every tool takes the account as either the channel name or the address, and a substring of either one is enough. Nested folders are listed as `parent/child`.

Then point your MCP client at `server.py`. In Claude Code that is an entry in the project's `.mcp.json`:

```json
{
  "mcpServers": {
    "mbsync": {
      "command": "python3",
      "args": ["/path/to/mbsync-mcp/server.py"]
    }
  }
}
```

Restart the client after editing the file.

### Credentials for the write tools

Reading needs no credentials. The write tools open an IMAP connection, and the password is looked up first in the system keyring, under the service name `mbsync-mcp`:

```bash
python -c "import keyring; keyring.set_password('mbsync-mcp', 'user@gmail.com', 'app-password')"
```

With no keyring entry the server falls back to the `PassCmd` of that account in `~/.mbsyncrc` and runs it. For Gmail the password to store is an [app password](https://support.google.com/accounts/answer/185833), not the one you log in with.

## Tools

Reading, on the local Maildir:

- `list_accounts`: accounts found in `~/.mbsyncrc`
- `list_folders`: folders of an account, nested ones included
- `count_emails`: total and unread count for a folder
- `get_emails`: messages filtered by date range and read status, optionally syncing first
- `get_unread_emails`: the unread ones, newest first
- `get_email_details`: headers, body and list of attachments of one message
- `search_emails`: search over sender, subject, body, or all three
- `get_inbox_summary`: counts and the senders you hear from most
- `daily_briefing`: one day of mail, counted and listed

Writing, over IMAP:

- `sync_account`: runs mbsync for one account, with a timeout
- `move_email`: moves a message between two folders
- `archive_email`: moves it to the account's archive folder, as the server names it
- `delete_email`: moves it to trash, likewise
- `cleanup_batch`: several deletions and archivings over a single connection

## Resources

Two MCP resources expose the configuration as JSON: `mbsync://accounts` for the account list, and `mbsync://account/<name>/folders` for the folders of each one.

## How messages are addressed

A message is identified by its position in a listing, `0` being the most recent. Reading orders the local Maildir by file time, while the write tools reopen the folder over IMAP and order it by message date. The two agree as long as the local copy is current, so run `sync_account` or re-read the folder before acting on an index that comes from an old listing.

## Smoke test

`test_simple.py` runs against the accounts your own `~/.mbsyncrc` declares: it parses the configuration, resolves each account by channel name and by address, and counts the messages in the folders it finds. It opens no IMAP connection and writes nothing.

```bash
python3 test_simple.py
```

## Debug

```bash
MBSYNC_DEBUG=1 python3 server.py
```

Debug lines go to standard output, which is also the channel the MCP client talks on, so use this while running the server by hand.

## License

MIT
