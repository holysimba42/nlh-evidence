# nlh-evidence

$0 no-human-in-the-loop credential automation, proven empirically.

## Layout
- `scripts/nlh.py` — the credential plane: `put` / `rotate` / `verify` /
  `oanda-fetch` / `oanda-validate` (see module docstring).
- `scripts/k3_totp_session.py` — proves unattended password+TOTP login,
  storageState session persistence, and expiry fail-loud (staging app).
- `scripts/k4_dom.py` — proves headless DOM navigate/extract/type/click.
- `.github/workflows/k5.yml` — consumes `BUF_TEST_SECRET` on the
  self-hosted runner (`nlh-local` label) and logs its sha256.
- `evidence/` — JSON proof files + screenshots from real runs.

## Daily cycle
Task Scheduler task `nlh-daily-rotate` fires 06:00 local:
`nlh.py rotate` seals a fresh secret via REST (libsodium sealed box),
dispatches the workflow, and verifies the runner-logged sha256 matches.
State: `evidence/last_rotate.json`. GitHub-hosted runners are blocked by
an account-level billing lock (`evidence/platform_gate/`); the
self-hosted runner bypasses it.

## Secrets never touch disk as plaintext
Values live in the OS keyring and GitHub Actions secrets. Only sha256
prefixes are recorded locally.
