"""Screenshot the locally running dashboard (http://localhost:8599) with the installed Edge browser.

usage: shot_dashboard.py OUT.png [generate]
With "generate": switch to "Browse Dataset Windows" (a real test alarm, so RCA evidence is injected) and
generate an explanation (one LLM call). On failure the current page is saved as OUT_debug.png.
"""
import sys

from playwright.sync_api import sync_playwright

OUT = sys.argv[1]
GENERATE = len(sys.argv) > 2 and sys.argv[2] == "generate"

with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    # tall viewport: Streamlit scrolls inside its own container, so full_page alone misses the lower panels
    page = browser.new_page(viewport={"width": 1440, "height": 2600}, color_scheme="light")
    try:
        page.goto("http://localhost:8599/", wait_until="networkidle")
        button = page.get_by_role("button", name="Generate Explanation")
        button.wait_for(timeout=400_000)
        if GENERATE:
            page.get_by_text("Browse Dataset Windows").click()
            page.wait_for_timeout(15_000)
            # filter to a fault with RCA evidence (Jamming windows have none)
            page.locator('[data-testid="stSelectbox"]').first.click()
            page.get_by_role("option", name="Doppler Shift (Severe)").click()
            page.wait_for_timeout(10_000)
            button.click(timeout=300_000)
            page.get_by_text("Recommended Action").first.wait_for(timeout=300_000)
            page.wait_for_timeout(5_000)
        page.screenshot(path=OUT, full_page=True)
        print("saved", OUT)
    except Exception as e:  # keep the evidence of what the page showed
        page.screenshot(path=OUT.replace(".png", "_debug.png"), full_page=True)
        print("failed:", type(e).__name__, str(e)[:300])
    finally:
        browser.close()
