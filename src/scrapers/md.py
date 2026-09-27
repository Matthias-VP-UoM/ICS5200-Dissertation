from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

# Utility function which attempts to close cookie/privacy pop-ups if they appear.
# If no pop-up is found, the script continues normally.
def close_popups_if_present(driver, timeout=5):
    possible_selectors = [
        (By.XPATH, '//*[@id="CybotCookiebotDialogBodyButtonDecline"]'), 
        (By.XPATH, '//*[@id="CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"]'),
        (By.XPATH, '//*[@aria-label="Close advertisement"]'),
        # (By.XPATH, "/html/body/div[1]/div[1]/div[1]/div[1]/div[2]/button[3]")
    ]

    for by, selector in possible_selectors:
        try:
            button = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((by, selector))
            )
            button.click()
            print("Pop-up closed.")
            time.sleep(1)
            return True

        except TimeoutException:
            continue

        except Exception as e:
            print(f"Pop-up selector found but could not click: {e}")
            continue

    print("No pop-up found. Continuing.")
    return False


def get_html_content(driver, site):
    driver.get(site)

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    popup_checker = close_popups_if_present(driver)

    if popup_checker == True:
        close_popups_if_present(driver)

    time.sleep(3)

    return driver.page_source