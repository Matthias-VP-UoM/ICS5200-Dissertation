import hashlib
import os
import time
import pandas as pd
from scrapers.tom import get_html_content as get_html_content_tom
from scrapers.nb import get_html_content as get_html_content_nb
from scrapers.ind import get_html_content as get_html_content_ind
from scrapers.mt import get_html_content as get_html_content_mt
from scrapers.ts import get_html_content as get_html_content_ts
from scrapers.md import get_html_content as get_html_content_md
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import json

def create_driver():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    return driver

def collect_dom(driver, url, output_path, web_source, wait_time=10) -> bool:
    try:
        if web_source == 'independent':
            html = get_html_content_ind(driver, url)
        elif web_source == 'malta_daily':
            html = get_html_content_md(driver, url)
        elif web_source == 'malta_today':
            html = get_html_content_mt(driver, url)
        elif web_source == 'newsbook':
            html = get_html_content_nb(driver, url)
        elif web_source == 'the_shift':
            html = get_html_content_ts(driver, url)
        elif web_source == 'times_of_malta':
            html = get_html_content_tom(driver, url)
        
        time.sleep(wait_time)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return True
    except Exception as e:
        print(f"Failed to collect DOM for {url}")
        print(f"Error: {e}")
        return False

def add_path_to_json(file_path, p_id, website, website_tracking_dict):
    if website not in website_tracking_dict:
        website_tracking_dict[website] = {'html_paths': {}}
    website_tracking_dict[website]['html_paths'][p_id] = file_path

def save_paths_file(website_tracking_dict, website):
    json_dir = os.path.join("data", "html", website)
    os.makedirs(json_dir, exist_ok=True)
    json_file_path = os.path.join(json_dir, 'html_paths.json')

    # Extract only this website's data to avoid mixing paths
    data_to_save = website_tracking_dict.get(website, {'html_paths': {}})

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)

def main():
    manifest_path = "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    driver = create_driver()

    processed_urls = set()
    
    website_tracking_dict = {}

    for _, row in manifest.iterrows():
        page_id = str(row["id"]).zfill(4)
        url = row["url"]
        website = row["website"]
        page_type = str(row["page_type"]).lower().replace(' ', '_')

        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:10]
        html_filename = f'{page_type}_{url_hash}.html'
        html_path = os.path.join("data", "html", website, html_filename)

        if url in processed_urls:
            print(f"[{page_id}] URL is a duplicate in the manifest. Skipping.")
            add_path_to_json(html_filename, page_id, website, website_tracking_dict)
            save_paths_file(website_tracking_dict, website)
            continue

        if os.path.exists(html_path):
            print(f"[{page_id}] HTML file already exists on disk. Skipping.")
            add_path_to_json(html_filename, page_id, website, website_tracking_dict)
            save_paths_file(website_tracking_dict, website)
            processed_urls.add(url)
            continue

        print(f"[{page_id}] Collecting DOM from: {url}")
        success = collect_dom(driver, url, html_path, website)

        if success:
            print(f"[{page_id}] Saved to {html_path}")
            processed_urls.add(url)
            add_path_to_json(html_filename, page_id, website, website_tracking_dict)
            save_paths_file(website_tracking_dict, website)
        else:
            print(f"[{page_id}] Failed.")
        
        time.sleep(2)

    driver.quit()

if __name__ == "__main__":
    main()