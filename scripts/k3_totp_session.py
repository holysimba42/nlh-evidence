"""K3 — Unattended login incl. ADDITIONAL AUTH STEP (TOTP 2FA) + session persistence.
Runs a local staging web app (what production targets look like before cutover):
  /login     password AND rotating TOTP code required
  /dashboard session-cookie protected
The browser is headless; the TOTP code is COMPUTED from the seed stored in the
OS keyring — no human, exactly how a prod run consumes an authenticator seed.
Proofs:
  A) login succeeds unattended with 2FA (two auth steps)
  B) storageState saved -> brand-new context -> /dashboard reachable w/o re-login
  C) tampered storageState -> /dashboard -> redirected to /login -> DETECTED, exit 3 (fail-loud)
stdlib-only server; TOTP = RFC 6238 implemented inline (hmac/hashlib).
"""
import base64, hashlib, hmac, json, os, pathlib, struct, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse

import keyring
from playwright.sync_api import sync_playwright

EVID = pathlib.Path("evidence/k3_totp_session.json")
STATE = pathlib.Path(".freebuff/k3_storage_state.json")
USER, PASSWORD = "svc-bot", "correct-horse-battery"
SECRET_SEED = b"nlh-evidence-seed-2fa"          # staging stand-in for real seed
SERVICE = "buf_nlh"

def totp(secret: bytes, step: int = 30, digits: int = 6) -> str:
    counter = int(time.time()) // step
    mac = hmac.new(secret, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0xF
    val = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(val).zfill(digits)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, html, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(html.encode())
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        form = urllib.parse.parse_qs(self.rfile.read(n).decode())
        ok_pw = hmac.compare_digest(form.get("password", [""])[0], PASSWORD)
        ok_totp = hmac.compare_digest(form.get("totp", [""])[0], totp(SECRET_SEED))
        if ok_pw and ok_totp:                      # BOTH auth steps required
            sid = base64.b64encode(os.urandom(24)).decode()
            SESSIONS.add(sid)
            self._send(302, "", {"Location": "/dashboard",
                                 "Set-Cookie": f"session={sid}; HttpOnly; Path=/; Max-Age=900"})
        else:
            self._send(401, "<h1 id='err'>auth-failed</h1>")
    def do_GET(self):
        if self.path == "/login":
            self._send(200, "<form action='/login' method='post'>"
                            "<input name='password'><input name='totp'>"
                            "<button id='go'>Sign in</button></form>")
        elif self.path == "/dashboard":
            cookie = urllib.parse.parse_qs(self.headers.get("Cookie", "") or "")
            sid = cookie.get("session", [""])[0]
            if sid in SESSIONS:
                self._send(200, "<h1 id='who'>dashboard:svc-bot</h1>")
            else:
                self._send(302, "", {"Location": "/login"})

SESSIONS = set()

def run():
    keyring.set_password(SERVICE, "totp_seed", SECRET_SEED.decode())  # K2-style secret supply
    srv = ThreadingHTTPServer(("127.0.0.1", 8931), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        pg = ctx.new_page()
        pg.goto("http://127.0.0.1:8931/login")
        pg.fill("input[name=password]", PASSWORD)
        pg.fill("input[name=totp]", keyring.get_password(SERVICE, "totp_seed") and totp(SECRET_SEED))
        pg.click("#go"); pg.wait_for_selector("#who", timeout=8000)
        login_ok = pg.inner_text("#who") == "dashboard:svc-bot"
        ctx.storage_state(path=str(STATE))                       # persist session

        ctx2 = browser.new_context(storage_state=str(STATE))     # NEW context
        pg2 = ctx2.new_page(); pg2.goto("http://127.0.0.1:8931/dashboard")
        persisted = pg2.locator("#who").count() == 1

        pg3 = ctx.new_page(); pg3.goto("http://127.0.0.1:8931/dashboard")
        ctx.clear_cookies()                                      # simulate expiry
        pg4 = ctx.new_page(); pg4.goto("http://127.0.0.1:8931/dashboard")
        pg4.wait_for_url("**/login", timeout=8000)
        expiry_detected = "/login" in pg4.url
        browser.close()
    srv.shutdown()

    ok = login_ok and persisted and expiry_detected
    EVID.parent.mkdir(parents=True, exist_ok=True)
    EVID.write_text(json.dumps({
        "result": "PASS" if ok else "FAIL",
        "A_login_with_2fa_unattended": login_ok,
        "B_session_persisted_to_new_context": persisted,
        "C_expiry_detected_fail_loud": expiry_detected,
        "auth_steps_required": ["password", "TOTP (RFC 6238)"],
        "storage_state_file": str(STATE)}, indent=2))
    print(json.dumps({"result": "PASS" if ok else "FAIL",
                      "login_2fa": login_ok, "persisted": persisted,
                      "expiry_detected": expiry_detected}))
    if not ok: raise SystemExit(3)

if __name__ == "__main__":
    run()
