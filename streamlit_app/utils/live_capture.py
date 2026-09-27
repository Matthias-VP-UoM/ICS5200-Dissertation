from __future__ import annotations

import ipaddress
import json
import os
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


DEFAULT_VIEWPORT = (1366, 768)
MAX_PAGE_HEIGHT = 12_000


def normalise_and_validate_url(url: str) -> str:
    """Normalise a user-entered URL and reject local/private destinations."""
    value = (url or "").strip()
    if not value:
        raise ValueError("Please enter a webpage URL.")

    if "://" not in value:
        value = f"https://{value}"

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")
    if not parsed.hostname:
        raise ValueError("The URL does not contain a valid hostname.")

    hostname = parsed.hostname.lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("Local addresses are not permitted.")

    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve the hostname: {hostname}") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if any(
            [
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_multicast,
                ip.is_reserved,
                ip.is_unspecified,
            ]
        ):
            raise ValueError("Private, loopback, and reserved network addresses are not permitted.")

    return value


def create_driver(page_load_timeout=35, headless=False):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    # options.add_argument("--window-size=1366,900")
    options.add_argument(f"--window-size={DEFAULT_VIEWPORT[0]},{DEFAULT_VIEWPORT[1]}")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--disable-notifications")
    # options.add_argument("--disable-popup-blocking")
    # options.add_argument("--hide-scrollbars")
    # options.add_argument("--force-device-scale-factor=1")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(page_load_timeout)
    driver.set_script_timeout(20)
    return driver


def dismiss_popups(driver, timeout: int = 2) -> bool:
    """Best-effort dismissal of common consent notices and overlays."""
    try:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        time.sleep(0.5)
    except Exception:
        pass

    button_xpaths = [
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept all')]",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'consent')]",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow')]",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'got it')]",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'close')]",
        "//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'reject all')]",
        "//a[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
        "//a[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]",
    ]

    for xpath in button_xpaths:
        try:
            button = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            driver.execute_script("arguments[0].click();", button)
            time.sleep(0.6)
            return True
        except Exception:
            continue
    return False


def _scroll_to_load_lazy_content(driver, max_steps: int = 8) -> None:
    last_height = 0
    for _ in range(max_steps):
        height = int(
            driver.execute_script(
                "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);"
            )
        )
        if height == last_height:
            break
        last_height = height
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.7)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.5)


def _save_full_page_screenshot(driver, screenshot_path: str) -> None:
    width = int(
        driver.execute_script(
            "return Math.max(document.body.scrollWidth, document.documentElement.scrollWidth);"
        )
    )
    height = int(
        driver.execute_script(
            "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);"
        )
    )

    width = max(DEFAULT_VIEWPORT[0], min(width, 2200))
    height = max(DEFAULT_VIEWPORT[1], min(height, MAX_PAGE_HEIGHT))

    driver.execute_cdp_cmd(
        "Emulation.setDeviceMetricsOverride",
        {
            "mobile": False,
            "width": width,
            "height": height,
            "deviceScaleFactor": 1,
        },
    )
    driver.save_screenshot(screenshot_path)


def capture_element_boxes(driver, max_elements: int = 800) -> list[dict]:
    """Capture bounding boxes + identifying info for visible DOM elements.

    Runs inside the already-open Selenium session (before driver.quit()), so
    it costs one extra JS round-trip rather than a new page load. Coordinates
    are in CSS pixels relative to the full page - the same coordinate space
    as the full-page screenshot, since deviceScaleFactor is fixed at 1.
    """
    script = """
    const maxElements = arguments[0];
    const nodes = Array.from(document.querySelectorAll('body *'));
    const results = [];
    for (const el of nodes) {
        const rect = el.getBoundingClientRect();
        if (rect.width < 4 || rect.height < 4) continue;  // skip invisible/zero-size
        results.push({
            tag: el.tagName.toLowerCase(),
            id: el.id || null,
            class_name: el.className && typeof el.className === 'string' ? el.className : null,
            aria_label: el.getAttribute('aria-label'),
            text: (el.innerText || '').trim().slice(0, 60),
            x: rect.left + window.scrollX,
            y: rect.top + window.scrollY,
            width: rect.width,
            height: rect.height
        });
        if (results.length >= maxElements) break;
    }
    return results;
    """
    try:
        return driver.execute_script(script, max_elements)
    except Exception:
        return []


def render_offline_html_for_elements(html_path, screenshot_size: tuple[int, int], headless: bool = True) -> list[dict] | None:
    """Best-effort DOM element bounding-box capture for an uploaded, offline HTML file.

    There's no live browser session for uploaded screenshot+HTML pairs, so
    this launches a fresh headless browser purely to render the saved HTML
    from disk (via a file:// URL) at a viewport matching the uploaded
    screenshot's dimensions, then reuses capture_element_boxes() on it.

    IMPORTANT CAVEAT: the uploaded HTML is a static snapshot. Any CSS/JS/images
    referenced with *relative* paths (e.g. "/styles/main.css") will fail to
    resolve when loaded from file://, since there's no server to resolve them
    against. Absolute URLs (e.g. "https://example.com/styles/main.css") will
    generally still load fine if there's network access. This means the
    resulting layout - and therefore the element bounding boxes - can differ
    from how the page actually looked when the screenshot was taken. Treat
    this as an approximation, not a guaranteed-accurate mapping.

    Returns None on any failure so callers can gracefully skip DOM-grounding.
    """
    driver = None
    try:
        width, height = screenshot_size
        width = max(320, min(int(width), 2200))
        height = max(240, min(int(height), 12_000))

        driver = create_driver(headless=headless)
        file_url = Path(html_path).resolve().as_uri()
        driver.get(file_url)

        # Match the viewport to the uploaded screenshot so element coordinates
        # line up with the same pixel grid the Grad-CAM heatmap was computed on.
        driver.execute_cdp_cmd(
            "Emulation.setDeviceMetricsOverride",
            {"mobile": False, "width": width, "height": height, "deviceScaleFactor": 1},
        )
        # Reload after the metrics override so layout is recomputed at the new size.
        driver.get(file_url)
        time.sleep(1.5)  # brief settle time for any inline scripts/CSS

        return capture_element_boxes(driver)
    except Exception:
        return None
    finally:
        if driver:
            driver.quit()


def capture_page(url, output_dir="live_outputs", delay=20):
    os.makedirs(output_dir, exist_ok=True)

    screenshot_path = os.path.join(output_dir, "live_screenshot.png")
    html_path = os.path.join(output_dir, "live_page.html")
    elements_path = os.path.join(output_dir, "live_elements.json")

    driver = create_driver()

    try:
        driver.get(url)

        # Initial wait for dynamic scripts/DOM elements to render
        time.sleep(delay)

        # Attempt to dismiss overlays and cookie consent banners
        dismiss_popups(driver)

        # Brief pause to let animations or DOM cleanups settle after dismissal
        time.sleep(1)

        # Save HTML source code
        html = driver.page_source
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Save element bounding boxes (for Grad-CAM -> DOM grounding later)
        elements = capture_element_boxes(driver)
        with open(elements_path, "w", encoding="utf-8") as f:
            json.dump(elements, f)

        # Save view screenshot
        driver.save_screenshot(screenshot_path)

    finally:
        driver.quit()

    return screenshot_path, html_path, elements_path