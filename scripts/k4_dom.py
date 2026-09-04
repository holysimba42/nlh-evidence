"""K4 — Full DOM surface on live github.com (ToS-tolerant read paths only).
navigate -> extract -> click -> navigate -> type -> assert -> screenshots.
"""
import json, pathlib
from playwright.sync_api import sync_playwright

EVID = pathlib.Path("evidence/k4_dom.json")

def main():
    shots = pathlib.Path("evidence/shots"); shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 800})

        # 1) navigate + extract + screenshot
        pg.goto("https://github.com/trending", wait_until="domcontentloaded", timeout=45000)
        first_repo = pg.locator("h2 a").first.inner_text().strip()
        pg.screenshot(path=str(shots / "k4_trending.png"))

        # 2) click first repo link -> assert landing URL (click-through proof)
        pg.locator("h2 a").first.click()
        pg.wait_for_load_state("domcontentloaded")
        landed_url = pg.url
        repo_title = pg.title()
        pg.screenshot(path=str(shots / "k4_repo.png"))

        # 3) type into a stable live form field (login page; NEVER submitted)
        pg.goto("https://github.com/login", wait_until="domcontentloaded", timeout=45000)
        pg.wait_for_selector("#login_field", timeout=45000)
        pg.fill("#login_field", "nlh-evidence-typed")
        typed_ok = pg.input_value("#login_field") == "nlh-evidence-typed"
        pg.screenshot(path=str(shots / "k4_login_typed.png"))
        b.close()

    norm = lambda s: s.replace(" ", "").replace("/", "").lower()
    ok = bool(first_repo) and norm(first_repo) in norm(landed_url) and typed_ok
    EVID.write_text(json.dumps({
        "result": "PASS" if ok else "FAIL",
        "navigate": ["github.com/trending", "repo page", "github.com/login"],
        "extracted_first_trending_repo": first_repo,
        "clicked_link_landed_url": landed_url,
        "repo_page_title": repo_title,
        "typed_and_verified_field": "#login_field",
        "typed_value_asserted": typed_ok,
        "screenshots": ["evidence/shots/k4_trending.png", "evidence/shots/k4_repo.png",
                         "evidence/shots/k4_login_typed.png"],
        "headless": True}, indent=2))
    print(json.dumps({"result": "PASS" if ok else "FAIL",
                      "extract": first_repo, "click_landed": landed_url,
                      "typed_ok": typed_ok}))
    if not ok: raise SystemExit(3)

if __name__ == "__main__":
    main()
