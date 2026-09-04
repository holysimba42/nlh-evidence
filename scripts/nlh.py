"""nlh — no-human-in-the-loop credential plane.
Responsibilities: resolve credential, seal secrets into GitHub Actions,
prove unattended consumption (dispatch -> self-hosted runner -> hash round-trip),
fetch/validate the OANDA token.

Subcommands:
  put NAME [VALUE]   seal value (arg or stdin) into GitHub secret NAME
  rotate             fresh BUF_TEST_SECRET -> put -> dispatch -> sha256 round-trip
  verify             re-dispatch -> log sha must match evidence/last_rotate.json
  oanda-fetch        stdin: agent-browser `read` DOM text -> keyring -> GitHub -> validate
  oanda-validate     keyring OANDA token -> GET /v3/accounts
  device-init        start OAuth device flow (RFC 8628): prints user code + URL
  device-poll        poll once (bounded 25-min window) -> keyring + scopes + GET /user proof
"""
import base64, hashlib, io, json, os, pathlib, re, subprocess, sys, time, urllib.request
import urllib.parse
import zipfile

import keyring
import nacl.public

REPO = "holysimba42/nlh-evidence"
WF = "k5.yml"
SECRET = "BUF_TEST_SECRET"
SERVICE = "buf_nlh"
EVID = pathlib.Path(__file__).resolve().parent.parent / "evidence"
ROT_STATE = EVID / "last_rotate.json"
SHA_RE = re.compile(r"secret_sha256=([0-9a-f]{64})")
OANDA_TOKEN_RE = re.compile(r"\b[0-9a-f]{32}-[0-9a-f]{32}\b")

CLIENT_ID = "178c6fc778ccc68e1d6a"          # GitHub CLI's public OAuth app
DEVICE_SCOPES = "repo workflow read:org"
DEVICE_STATE = pathlib.Path.home() / ".nlh" / "device_state.json"
DEVICE_EVID = EVID / "device_flow.json"


def resolve_token(plane):
    """Single owner of credential resolution.
    gh plane: device-flow token (keyring) primary, gh CLI store fallback.
    oanda plane: keyring entry only — absence must surface, never mask."""
    if plane == "gh":
        tok = keyring.get_password(SERVICE, "gh_device_token")
        if tok:
            return tok
        tok = subprocess.run(["gh", "auth", "token"], capture_output=True,
                             text=True).stdout.strip()
        assert tok, "no gh credential in keyring or gh store"
        return tok
    if plane == "oanda":
        tok = keyring.get_password(SERVICE, "oanda_practice_token")
        assert tok, "no OANDA token in keyring"
        return tok
    raise ValueError(f"unknown plane: {plane}")


_GH_ENV = None


def _gh_env():
    global _GH_ENV
    if _GH_ENV is None:
        _GH_ENV = {**os.environ, "GH_TOKEN": resolve_token("gh")}
    return _GH_ENV


def gh(*args, binary=False):
    r = subprocess.run(["gh", "api", *args], capture_output=True, timeout=300,
                       env=_gh_env())
    assert r.returncode == 0, f"gh {args[:3]} failed: {r.stderr[:200]}"
    return r.stdout if binary else r.stdout.decode()


def seal_into_github(name, value):
    """Sealed-box encrypt value and store as GitHub Actions secret NAME."""
    pk = json.loads(gh(f"repos/{REPO}/actions/secrets/public-key"))
    sealed = base64.b64encode(nacl.public.SealedBox(
        nacl.public.PublicKey(base64.b64decode(pk["key"]))).encrypt(value.encode())).decode()
    gh("-X", "PUT", f"repos/{REPO}/actions/secrets/{name}",
       "-f", f"encrypted_value={sealed}", "-f", f"key_id={pk['key_id']}")


def dispatch_and_wait():
    gh("-X", "POST", f"repos/{REPO}/actions/workflows/{WF}/dispatches", "-f", "ref=main")
    time.sleep(8)
    for _ in range(48):
        time.sleep(5)
        runs = json.loads(gh(f"repos/{REPO}/actions/workflows/{WF}/runs?per_page=1"))
        r0 = runs["workflow_runs"][0]
        if r0["status"] == "completed":
            return r0
    raise SystemExit("run never completed")


def cloud_secret_sha(run_id):
    """sha256 the runner logged for the injected secret, from the run log zip."""
    p = subprocess.run(["gh", "api", f"repos/{REPO}/actions/runs/{run_id}/logs"],
                       capture_output=True, timeout=300, env=_gh_env())
    assert p.returncode == 0 and p.stdout[:2] == b"PK", "log zip fetch failed"
    with zipfile.ZipFile(io.BytesIO(p.stdout)) as z:
        text = "\n".join(z.read(n).decode(errors="replace")
                         for n in z.namelist() if n.endswith(".txt"))
    m = SHA_RE.search(text)
    return m.group(1) if m else None


def cmd_put(name, value=None):
    if value is None:
        value = sys.stdin.read().strip()
    assert value, "no value given"
    seal_into_github(name, value)
    print(json.dumps({"result": "PASS", "secret": name,
                      "sha256_prefix": hashlib.sha256(value.encode()).hexdigest()[:12]}))


def cmd_rotate():
    """Rotate secret, prove consumption, and drift-check the OANDA token.
    exit 3 = hash round-trip failure; exit 4 = OANDA drift (dead/rotted token).
    State (incl. failures) is written BEFORE exiting — never silent rot."""
    value = f"NLH-{int(time.time())}"
    seal_into_github(SECRET, value)
    local_sha = hashlib.sha256(value.encode()).hexdigest()
    run = dispatch_and_wait()
    cloud_sha = cloud_secret_sha(run["id"])
    rotate_ok = cloud_sha == local_sha and run["conclusion"] == "success"

    o_st, o_n = 0, 0
    try:
        o_st, o_n = oanda_accounts(resolve_token("oanda"))
    except AssertionError:
        pass                                   # no token == drift, recorded below
    oanda_ok = o_st == 200 and o_n > 0

    EVID.mkdir(exist_ok=True)
    ROT_STATE.write_text(json.dumps({
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "value_sha256": local_sha, "run_id": run["id"],
        "cloud_sha256": cloud_sha,
        "rotate_result": "PASS" if rotate_ok else "FAIL",
        "oanda_http_status": o_st, "oanda_accounts": o_n,
        "oanda_result": "PASS" if oanda_ok else "FAIL"}, indent=2))
    print(json.dumps({"rotate": "PASS" if rotate_ok else "FAIL",
                      "oanda": "PASS" if oanda_ok else "FAIL",
                      "oanda_http": o_st, "run_id": run["id"],
                      "url": run["html_url"]}))
    if not rotate_ok:
        raise SystemExit(3)
    if not oanda_ok:
        raise SystemExit(4)


def cmd_verify():
    last = json.loads(ROT_STATE.read_text())
    run = dispatch_and_wait()
    cloud_sha = cloud_secret_sha(run["id"])
    ok = cloud_sha == last["value_sha256"] and run["conclusion"] == "success"
    print(json.dumps({"result": "PASS" if ok else "FAIL", "run_id": run["id"],
                      "sha_match": cloud_sha == last["value_sha256"]}))
    if not ok:
        raise SystemExit(3)


def oanda_accounts(token):
    req = urllib.request.Request("https://api-fxpractice.oanda.com/v3/accounts",
                                 headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, len(json.loads(r.read().decode()).get("accounts", []))
    except urllib.error.HTTPError as e:
        return e.code, 0


def cmd_oanda_fetch():
    m = OANDA_TOKEN_RE.search(sys.stdin.read())
    if not m:
        print(json.dumps({"result": "FAIL", "stage": "token_not_found"}))
        raise SystemExit(3)
    tok = m.group(0)
    keyring.set_password(SERVICE, "oanda_practice_token", tok)   # encrypted at rest
    seal_into_github("OANDA_PRACTICE_TOKEN", tok)
    st, n = oanda_accounts(tok)
    (EVID / "oanda_fetch.json").write_text(json.dumps({
        "result": "PASS" if st == 200 and n else "FAIL", "http_status": st,
        "accounts_found": n, "stored": "keyring + GitHub secret",
        "printed_plaintext": False}, indent=2))
    print(json.dumps({"result": "PASS" if st == 200 and n else "FAIL",
                      "http_status": st, "accounts_found": n}))
    if st != 200 or not n:
        raise SystemExit(3)


def cmd_oanda_validate():
    st, n = oanda_accounts(resolve_token("oanda"))
    print(json.dumps({"result": "PASS" if st == 200 and n else "FAIL",
                      "http_status": st, "accounts_found": n}))
    if st != 200 or not n:
        raise SystemExit(3)


def _device_post(url, data):
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def cmd_device_init():
    """RFC 8628 §3.1-3.2: request device + user codes, persist the window."""
    d = _device_post("https://github.com/login/device/code",
                     {"client_id": CLIENT_ID, "scope": DEVICE_SCOPES})
    DEVICE_STATE.parent.mkdir(parents=True, exist_ok=True)
    d["initiated_at"] = time.time()
    DEVICE_STATE.write_text(json.dumps(d))
    print("=" * 50)
    print("  OPEN:  " + d["verification_uri"])
    print("  CODE:  " + d["user_code"])
    print("=" * 50)


def cmd_device_poll():
    """One bounded pass: poll the token endpoint until granted, denied, or the
    25-minute window closes. exit 2 on terminal errors — no silent hangs."""
    d = json.loads(DEVICE_STATE.read_text())
    deadline = d["initiated_at"] + 1500
    interval = d["interval"]
    tok = None
    while time.time() < deadline:
        time.sleep(min(interval, max(1, deadline - time.time())))
        r = _device_post("https://github.com/login/oauth/access_token",
                         {"client_id": CLIENT_ID, "device_code": d["device_code"],
                          "grant_type": "urn:ietf:params:oauth:grant-type:device_code"})
        if "access_token" in r:
            tok = r["access_token"]; break
        err = r.get("error")
        if err == "authorization_pending":
            continue
        if err == "slow_down":                  # RFC 8628 §3.5: retry slower
            interval += 5
            print(json.dumps({"event": "slow_down", "interval_s": interval}), flush=True)
            continue
        print(json.dumps({"result": "FAIL", "error": err or "unknown"}))
        raise SystemExit(2)
    if not tok:
        print(json.dumps({"result": "FAIL", "error": "expired_token_or_window"}))
        raise SystemExit(2)

    # proof: authenticated GET /user using ONLY the device-flow token
    req = urllib.request.Request("https://api.github.com/user", headers={
        "Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
        "User-Agent": "nlh-evidence"})
    with urllib.request.urlopen(req, timeout=30) as r:
        user = json.loads(r.read().decode())
    scopes = [s for s in r.get("scope", "").split(",") if s]   # from token response
    keyring.set_password(SERVICE, "gh_device_token", tok)   # distinct entry
    EVID.mkdir(exist_ok=True)
    DEVICE_EVID.write_text(json.dumps({
        "result": "PASS", "login": user["login"], "user_id": user["id"],
        "get_user_status": 200, "scopes": scopes,
        "token_prefix": tok[:4] + "****",
        "token_sha256_prefix": hashlib.sha256(tok.encode()).hexdigest()[:12],
        "stored": "OS keyring (buf_nlh/gh_device_token)",
        "token_printed_anywhere": False}, indent=2))
    print(json.dumps({"result": "PASS", "login": user["login"],
                      "scopes": scopes, "stored": "keyring:gh_device_token"}))


USAGE = """usage: nlh.py <command> [args]
commands:
  put NAME [VALUE]   seal value (arg or stdin) into GitHub secret NAME
  rotate             fresh secret -> dispatch -> sha256 round-trip + OANDA drift check
  verify             re-dispatch; log sha must match last rotate
  oanda-fetch        stdin: DOM text -> keyring + GitHub secret -> validate
  oanda-validate     keyring OANDA token -> GET /v3/accounts
  device-init        start OAuth device flow (prints code + URL)
  device-poll        bounded poll -> keyring + GET /user proof
"""


def main():
    cmds = {"put": cmd_put, "rotate": cmd_rotate, "verify": cmd_verify,
            "oanda-fetch": cmd_oanda_fetch, "oanda-validate": cmd_oanda_validate,
            "device-init": cmd_device_init, "device-poll": cmd_device_poll}
    if len(sys.argv) < 2:
        print(USAGE, end="")
        return
    fn = cmds.get(sys.argv[1])
    if fn is None:
        print(f"unknown command: {sys.argv[1]}", file=sys.stderr)
        raise SystemExit(2)
    fn(*sys.argv[2:])


if __name__ == "__main__":
    main()
