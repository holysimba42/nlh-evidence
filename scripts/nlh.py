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
"""
import base64, hashlib, io, json, pathlib, re, subprocess, sys, time, urllib.request
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


def gh(*args, binary=False):
    r = subprocess.run(["gh", "api", *args], capture_output=True, timeout=300)
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
                       capture_output=True, timeout=300)
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
    value = f"NLH-{int(time.time())}"
    seal_into_github(SECRET, value)
    local_sha = hashlib.sha256(value.encode()).hexdigest()
    run = dispatch_and_wait()
    cloud_sha = cloud_secret_sha(run["id"])
    ok = cloud_sha == local_sha and run["conclusion"] == "success"
    EVID.mkdir(exist_ok=True)
    ROT_STATE.write_text(json.dumps({"value_sha256": local_sha, "run_id": run["id"],
                                     "cloud_sha256": cloud_sha,
                                     "result": "PASS" if ok else "FAIL"}))
    print(json.dumps({"result": "PASS" if ok else "FAIL", "run_id": run["id"],
                      "sha_match": cloud_sha == local_sha, "url": run["html_url"]}))
    if not ok:
        raise SystemExit(3)


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
    tok = keyring.get_password(SERVICE, "oanda_practice_token")
    assert tok, "no OANDA token in keyring"
    st, n = oanda_accounts(tok)
    print(json.dumps({"result": "PASS" if st == 200 and n else "FAIL",
                      "http_status": st, "accounts_found": n}))
    if st != 200 or not n:
        raise SystemExit(3)


if __name__ == "__main__":
    cmds = {"put": cmd_put, "rotate": cmd_rotate, "verify": cmd_verify,
            "oanda-fetch": cmd_oanda_fetch, "oanda-validate": cmd_oanda_validate}
    cmds[sys.argv[1]](*sys.argv[2:])
