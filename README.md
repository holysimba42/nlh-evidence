# nlh-evidence

$0 no-human-in-the-loop credential automation, proven empirically.

## Layout
- `scripts/nlh.py` — the credential plane; `oanda_token()` resolves the
  OANDA credential (keyring only — absence surfaces as exit 4, never
  masked); gh calls use the gh CLI's own credential store.
  Commands: `put` / `rotate` / `verify` / `oanda-fetch` / `oanda-validate` /
  `device-init` / `device-poll`.
- `scripts/k3_totp_session.py` — proves unattended password+TOTP login,
  storageState session persistence, and expiry fail-loud (staging app).
- `scripts/k4_dom.py` — proves headless DOM navigate/extract/type/click.
- `.github/workflows/k5.yml` — consumes `BUF_TEST_SECRET` on the
  self-hosted runner (`nlh-local` label) and logs its sha256.
- `setup.ps1` — idempotent bootstrap: deps, runner registration, daily task.

## Drift-detection contract
Every `rotate` (also the 06:00 scheduled one) runs BOTH checks and writes
`evidence/last_rotate.json` (untracked mutable state) with fields:
`checked_at`, `value_sha256`, `run_id`, `cloud_sha256`, `rotate_result`,
`oanda_http_status`, `oanda_accounts`, `oanda_result`.

- exit 0 — all green
- exit 3 — hash drift (runner-observed secret ≠ sealed value)
- exit 4 — OANDA drift (`oanda_http_status != 200`, or no token/zero accounts)
- exit 5 — infrastructure failure (dispatch/log); state records the error

Failures are recorded in state BEFORE exit — no silent rot. Historical
fail-loud proofs: the real OANDA 401 drift episode is visible in the
conversation log; `expired_token` device-flow exits are reproduced by
`nlh device-poll` whenever the 15-minute code window lapses.

## Schedule proof
`evidence/schedule_proof.json` — Windows Task Scheduler (not manual
dispatch) fired the full chain: task fired 05:53:00, workflow run
33875046897 created 4s later, rotate PASS + OANDA PASS recorded. The
production task `nlh-daily-rotate` fires daily at 06:00 with the identical
command. GitHub-hosted runners are blocked by an account-level billing
lock (`evidence/platform_gate/`); the self-hosted runner bypasses it.

## Known blocks (honest state)
- **K1 device-flow mint**: implemented (`nlh device-init && nlh
  device-poll`) and fail-loud-proven (RFC 8628 expiry/slow-down exits),
  but the final mint needs ONE human click at github.com/login/device.
  Four windows expired unclicked; no token was minted. gh-plane
  credentials currently come from the pre-existing gh CLI keyring store.
- **OANDA re-auth**: only ever needed if `oanda-validate`/`rotate` reports
  exit 4 (drift). Otherwise the stored token keeps validating 200.

## Secrets never touch disk as plaintext
Values live in the OS keyring and GitHub Actions secrets. Only sha256
prefixes are recorded locally.
