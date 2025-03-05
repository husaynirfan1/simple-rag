import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL of the website
BASE_URL = "https://english.almanar.com.lb"

# Sections to scrape
SECTIONS = {
    "news": "https://english.almanar.com.lb/cat/news",
    "exclusive": "https://english.almanar.com.lb/cat/exclusive",
    "speeches": "https://english.almanar.com.lb/cat/news/lebanon/s-nasrallah-speeches",
    "hezbollah_statements": "https://english.almanar.com.lb/cat/news/lebanon/hezbollah-statements"
}

# Function to fetch and parse a page
def fetch_page(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

# Function to extract article details
def extract_article_details(article_url):
    soup = fetch_page(article_url)
    if not soup:
        return None

    # Extract title
    title_tag = soup.find("div", class_="article-title")
    title = title_tag.find("h2").text.strip() if title_tag and title_tag.find("h2") else "No Title"

    # Extract date
    meta_tag = soup.find("div", class_="article-meta")
    date = meta_tag.find("span").text.strip() if meta_tag and meta_tag.find("span") else "No Date"

    # Extract content
    content_tag = soup.find("div", class_="article-content")
    if content_tag:
        content = "\n".join([p.text.strip() for p in content_tag.find_all("p") if p.text.strip()])
    else:
        content = "No Content"

    return {
        "url": article_url,
        "title": title,
        "date": date,
        "content": content
    }

# Function to scrape a section
def scrape_section(section_url, pages, save_path):
    all_articles = []
    for page in range(1, pages + 1):
        # Handle page URL logic
        if pages == 1:
            page_url = section_url  # Use the base URL for the first page
        else:
            page_url = f"{section_url}/page/{page}"  # Append /page/{page} for subsequent pages

        print(f"Scraping {page_url}")
        soup = fetch_page(page_url)
        if not soup:
            continue

        # Find articles in double-single div
        double_single = soup.find("div", class_="double-single")
        if double_single:
            for col in double_single.find_all("div", class_="col-sm-6"):
                for box in col.find_all("div", class_="single-box"):
                    links = box.find_all("a", href=True)
                    if len(links) > 1:
                        article_url = urljoin(BASE_URL, links[1]["href"])
                        article_details = extract_article_details(article_url)
                        if article_details:
                            all_articles.append(article_details)

        # Find articles in single-box divs below double-single
        single_boxes = soup.find_all("div", class_="single-box")
        for box in single_boxes:
            # Find the <a> tag with href inside the <div class="row">
            rows = box.find_all("div", class_="row")
            for row in rows:
                link = row.find("a", href=True)
                if link:
                    article_url = urljoin(BASE_URL, link["href"])
                    article_details = extract_article_details(article_url)
                    if article_details:
                        all_articles.append(article_details)
                    break  # Stop after finding the first valid link

    # Save articles to files
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for article in all_articles:
        file_name = f"{article['title'].replace('/', '_')}.txt"
        file_path = os.path.join(save_path, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {article['url']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Date: {article['date']}\n")
            f.write(f"Content:\n{article['content']}\n")

    return all_articles

# Main function
def main():
    # User input for sections to scrape
    print("Select sections to scrape (comma-separated):")
    for key in SECTIONS:
        print(f"- {key}")
    selected_sections = input("Enter section keys: ").strip().split(",")

    # User input for number of pages
    pages = int(input("Enter number of pages to scrape per section: "))

    # User input for save path
    save_path = input("Enter path to save scraped content: ").strip()

    # Scrape selected sections
    all_content = []
    for section in selected_sections:
        section = section.strip()
        if section in SECTIONS:
            section_url = SECTIONS[section]
            section_path = os.path.join(save_path, section)
            print(f"Scraping section: {section}")
            articles = scrape_section(section_url, pages, section_path)
            all_content.extend(articles)

    # Save all content to a single folder
    all_content_path = os.path.join(save_path, "all_content")
    if not os.path.exists(all_content_path):
        os.makedirs(all_content_path)

    for article in all_content:
        file_name = f"{article['title'].replace('/', '_')}.txt"
        file_path = os.path.join(all_content_path, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"URL: {article['url']}\n")
            f.write(f"Title: {article['title']}\n")
            f.write(f"Date: {article['date']}\n")
            f.write(f"Content:\n{article['content']}\n")

    print("Scraping completed!")

if __name__ == "__main__":
    main()
