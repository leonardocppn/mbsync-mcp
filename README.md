# mbsync-mcp

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides email access via [mbsync](https://isstracker.github.io/mbsync/) and Maildir.

## Features

- **Read emails** from Maildir folders synchronized with mbsync
- **Search** by sender, subject, or body content
- **Filter** by date range and read/unread status
- **IMAP operations**: move, delete, archive emails
- **Batch cleanup**: delete/archive multiple emails at once
- **Daily briefing**: inbox summary with top senders

## Requirements

- Python 3.10+
- [mbsync](https://isstracker.github.io/mbsync/) configured with `~/.mbsyncrc`
- [mcp](https://pypi.org/project/mcp/) >= 1.0.0
- [keyring](https://pypi.org/project/keyring/) (optional, for IMAP write operations)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mbsync-mcp.git
cd mbsync-mcp

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The server reads email configuration from `~/.mbsyncrc`. Make sure mbsync is properly configured and your emails are synchronized.

Example `~/.mbsyncrc` structure:
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

### IMAP Write Operations

To enable move/delete/archive operations, store your IMAP password:

```bash
# Using keyring
python -c "import keyring; keyring.set_password('mbsync-mcp', 'user@gmail.com', 'your-app-password')"
```

For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833).

## Usage with Claude Code

Add to your MCP configuration (e.g., `~/.config/claude/mcp.json`):

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

## Available Tools

### Read Operations
- `list_accounts` - List configured email accounts
- `list_folders` - List folders in an account
- `count_emails` - Count emails in a folder
- `get_emails` - Get emails with filters (date, read status)
- `get_unread_emails` - Get unread emails
- `get_email_details` - Get full email content
- `search_emails` - Search by sender/subject/body
- `get_inbox_summary` - Inbox statistics
- `daily_briefing` - Daily email summary

### Write Operations (require IMAP credentials)
- `sync_account` - Sync via mbsync
- `move_email` - Move email between folders
- `delete_email` - Move email to trash
- `archive_email` - Archive email
- `cleanup_batch` - Batch delete/archive

## Running Tests

```bash
cd mbsync-mcp
python test_simple.py
```

## Debug Mode

Enable debug output:
```bash
MBSYNC_DEBUG=1 python server.py
```

## License

MIT
