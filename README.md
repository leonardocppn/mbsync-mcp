# mbsync-mcp

A [Model Context Protocol](https://modelcontextprotocol.io/) server that gives an AI assistant access to the mail that [mbsync](https://isync.sourceforge.io/) keeps on your disk.

The read tools work on the local Maildir only, so they need no network connection. Moving, archiving and deleting are sent to the mail server as IMAP commands, and after each successful write the server runs mbsync for that account to update the local copy.

Accounts and their Maildir paths are read from `~/.mbsyncrc`. The server has no configuration file of its own, and reading mail needs no password.

## Requirements

Linux or macOS (on Windows, only under WSL), Python 3.10 or later, mbsync installed and configured, and `mcp>=1.0.0`. The `keyring` package is needed only to store the IMAP password in the system keyring.

## Installation

```bash
git clone https://github.com/leonardocppn/mbsync-mcp.git
cd mbsync-mcp
pip install -r requirements.txt
```

## Configuration

The server parses `~/.mbsyncrc` to find your accounts, so the accounts that mbsync syncs are the ones the tools can use. A minimal account looks like this.

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

Tools take the account as its full channel name or address, in upper or lower case. Partial names are rejected, because the same fragment can belong to more than one account and a write could reach the wrong mailbox. Shorter names can be defined as aliases (see below). Folders are found at any depth and listed as `parent/child`, and a tool accepts a folder name only if it appears in that list.

Then point your MCP client at `server.py`. In Claude Code this is an entry in the project's `.mcp.json`.

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

### Environment variables

These variables are optional and go in the `env` block of the same entry.

```json
"env": {
  "MBSYNC_ALIASES": "work=myaccount, me=user@gmail.com",
  "MBSYNC_MAX_BATCH": "20"
}
```

| Variable | Default | Meaning |
|---|---|---|
| `MBSYNC_ALIASES` | none | Short names for accounts, as `alias=account` pairs separated by commas, with the account given by channel name or address |
| `MBSYNC_READONLY` | off | With `1` or `true` the write tools are hidden and write requests are refused |
| `MBSYNC_MAX_BATCH` | `20` | Maximum number of messages moved by a single call |
| `MBSYNC_MAX_DAILY` | `200` | Maximum number of messages moved in the last 24 hours |
| `MBSYNC_JOURNAL` | `~/.local/state/mbsync-mcp/journal.jsonl` | Path of the write journal |
| `MBSYNC_DEBUG` | off | With `1` or `true` debug lines are printed to standard error |

The server skips an alias with a wrong format, an unknown account, a name equal to another account's name or address, or two different targets. The reason appears in the output of `list_accounts` and on standard error, and the server keeps running.

### Credentials for the write tools

The write tools open an IMAP connection and look up the password first in the system keyring, under the service name `mbsync-mcp`.

```bash
python -c "import keyring; keyring.set_password('mbsync-mcp', 'user@gmail.com', 'app-password')"
```

When the keyring has no entry for the account, the server runs the account's `PassCmd` from `~/.mbsyncrc` and waits up to 60 seconds for it. If neither source returns a password, the error message lists the reason for each. For Gmail the password to store is an [app password](https://support.google.com/accounts/answer/185833).

## Tools

Reading, on the local Maildir:

- `list_accounts` lists the accounts found in `~/.mbsyncrc`, with their aliases.
- `list_folders` lists the folders of an account, nested ones included.
- `count_emails` gives the total and unread count of a folder.
- `get_emails` lists messages filtered by date range and read status, optionally after a sync.
- `get_unread_emails` lists the unread messages, newest first.
- `get_email_details` shows the headers, the body and the attachment list of one message.
- `search_emails` searches sender, subject, the whole body, or all three, ignoring case.
- `get_inbox_summary` gives the counts and the most frequent senders among the latest 100 messages.
- `daily_briefing` gives the counts, the frequent senders and the unread messages, optionally only those that arrived from a given day on.

Writing, over IMAP:

- `sync_account` runs mbsync for one account, with a timeout.
- `move_email` moves a message from one folder to another.
- `archive_email` moves a message to the folder the mail server flags as the archive.
- `delete_email` moves a message to the folder the mail server flags as the trash.
- `cleanup_batch` deletes and archives several messages over a single connection.
- `list_operations` lists the latest writes recorded in the journal, each with its operation id.
- `undo_operation` puts the messages of an operation back in their folders, and with no id it undoes the latest operation.

Dates are shown in local time, and a `Date` header without a timezone is read as UTC. The filters `date_from` and `date_to` both include their whole day. When a message has only an HTML part, its body is converted to plain text. `get_email_details` prints the body under a line that labels it as external content, as a guard against instructions written into an email.

## Resources

The configuration is also available as two MCP resources in JSON. `mbsync://accounts` contains the account list without the `PassCmd`, and `mbsync://account/<name>/folders` contains the folders of one account.

## How messages are addressed

Listings show each message with an id in square brackets, and the tools take that id. The id is the message's UID on the IMAP server. mbsync keeps a state file in each folder (`.mbsyncstate`) that maps the local files to the server's UIDs, and mbsync-mcp reads the UIDs from there. A UID stays the same while the message remains in its folder, so mail that arrives after a listing does not change the ids shown in it.

Before moving a message, mbsync-mcp compares the folder's UIDVALIDITY on the IMAP server with the value recorded by mbsync. A different value means the server has renumbered the folder, and in that case nothing is written until `sync_account` updates the local state. mbsync-mcp then fetches the headers of the UID from the server and compares them with the local copy, using the `Message-ID` or, when there is none, subject, sender and date. After the move it searches the source folder for the UID, because an IMAP server answers OK to a MOVE even when the UID does not exist. A message that fails one of these checks stays where it is, and the result reports the reason.

## Safety of writes

The write tools run without asking for confirmation, and the server applies its own limits. Each successful move is recorded in a JSONL journal with the data needed to find the message again, and if the journal cannot be written the server moves nothing. `undo_operation` reads the journal and checks the identity of each message before moving it back. When more than one message in the folder matches the recorded identity, it leaves them where they are and reports how many it found. The limits on the number of messages and the read-only mode are set with the environment variables listed above. Tools run one at a time, so two writes, or two mbsync runs on the same Maildir, cannot overlap.

On Gmail folders are labels, and All Mail contains the messages outside Trash and Spam. Archiving from INBOX is a move to All Mail, which removes the INBOX label. For a move out of All Mail to a folder other than Trash, the server sends a COPY, which adds the label and keeps the message in All Mail. Deleting from All Mail is a move to Trash.

Folder names with non-ASCII characters, such as `Café`, are sent in the modified UTF-7 encoding that IMAP requires.

## Smoke test

`test_simple.py` runs against the accounts declared in your `~/.mbsyncrc`. It parses the configuration, resolves each account by channel name and by address, checks that a partial name is rejected and counts the messages in each folder, without opening IMAP connections or writing anything.

```bash
python3 test_simple.py
```

## Debug

```bash
MBSYNC_DEBUG=1 python3 server.py
```

Debug lines go to standard error, because the MCP protocol runs over standard output.

## License

MIT. See [LICENSE](LICENSE).
