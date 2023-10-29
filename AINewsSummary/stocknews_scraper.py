# Required modules and libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import requests
import requests_random_user_agent
from bs4 import BeautifulSoup
import sqlite3
from database import SentimentDatabase
import threading

# Initialization of the SentimentDatabase
db = SentimentDatabase()
db.drop_table()  # Drops the existing table if present
db.setup_db()    # Sets up the database

def click_accept_all_button(driver: webdriver.Chrome) -> None:
    """
    Click the "Accept All" button on a webpage if it exists.
    
    Args:
        driver (webdriver.Chrome): The Chrome driver instance.
    """
    try:
        # Locating and clicking the 'Accept All' and 'Allow All' buttons if they exist
        accept_all_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Accept All')]")
        accept_all_button.click()
        allow_all = driver.find_element(By.XPATH, "//button[contains(text(), 'Allow All')]")
        allow_all.click()
    except:
        pass  # Silently pass if the buttons aren't found or any error occurs

def load_webpage_with_random_user_agent(URL: str) -> webdriver.Chrome:
    """
    Loads a webpage with a random user-agent using Selenium.

    Args:
        URL (str): The URL of the webpage to load.

    Returns:
        webdriver.Chrome: An instance of the Chrome browser with the loaded webpage.
    """
    # Setting Chrome browser options
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    options.add_argument('log-level=3')
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--headless")

    # Fetching a random user-agent from httpbin
    resp = requests.get('https://httpbin.org/user-agent')
    random_agent = resp.json()['user-agent']
    options.add_argument(f"--user-agent={random_agent}")

    # Initializing the Chrome driver with the above-defined options
    driver = webdriver.Chrome(options=options)
    driver.get(URL)
    time.sleep(0.5)  # Pause for half a second

    # Handle special cases based on page title
    try:
        # Clicking on a specific element on the page (this seems to be a consent button)
        element = driver.find_element(By.XPATH,'/html/body/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[2]/div/div/button/span')
        time.sleep(1)
        element.click()

        # Handling behavior based on title conditions
        title = driver.title
        if 'Google News - Search' in title or 'Access to this page has been denied.' in title or len(title) <= 50 or '?' in title:
            return driver
        else:
            # If none of the above conditions are met, store the title
            db.insert_title_stock(stockname=None, title=title)  # Insert title with stockname as None
            return driver
    except Exception as e:
        # Print any exceptions that occur
        print(e)

    # Return the driver instance
    return driver

def open_links(soup) -> None:
    """
    Open links found in the provided soup object.

    Args:
        soup: A BeautifulSoup object containing the webpage content.
    """
    # Extract all hrefs from the soup object
    hrefs = [a['href'] for a in soup.find_all('a', class_='VDXfz') if a.has_attr('href')]
    print(f"found: {len(hrefs)} urls")

    # Open each href with a randomized user agent
    for url in hrefs:
        load_webpage_with_random_user_agent(f"https://news.google.com/articles/{str(url).replace('./articles/','')}")
        time.sleep(1)  # Pause for a second between each webpage load

def scroll_to_end(driver, times, wait_sec):
    """
    Scrolls a web page to the end a specified number of times.

    Args:
        driver: The Chrome driver instance.
        times (int): Number of times to scroll.
        wait_sec (int): Seconds to wait between each scroll.

    Returns:
        The Chrome driver instance after scrolling.
    """
    for x in range(times):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")  # Scroll to bottom
        time.sleep(wait_sec)  # Wait for specified time
    return driver

def backup_html(driver):
    """
    Saves the current page's HTML source to a local file.

    Args:
        driver: The Chrome driver instance.
    """
    with open('homepage.html', 'w+', encoding='utf-8') as e:
        e.write(driver.page_source)  # Write the page's HTML source to the file

def scrape_and_insert_titles(stockname: str, db: SentimentDatabase) -> None:
    """
    Scrape article titles for a specific stock and insert into the database.

    Args:
        stockname (str): Name of the stock.
        db (SentimentDatabase): Database object for inserting titles.
    """
    # Constructing the URL to scrape
    URL = f'https://news.google.com/search?q=stock+{stockname}'
    driver = load_webpage_with_random_user_agent(URL)
    driver.implicitly_wait(10)  # Implicit wait for elements to load

    # Parsing the webpage using BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    links = soup.find_all('a', class_='DY5T1d RZIKme')
    print(f"found {len(links)} articles for {stockname}")

    # Extract titles from the links and insert them into the database
    for link in links:
        title = link.text
        if "?" in title or len(title) < 49:
            continue
        else:
            db.insert_stock_title(stockname=stockname, title=title)  # Insert title with the specified stockname

    driver.quit()  # Close the browser instance

def main():
    """
    Main function to start the scraping process for specified stocknames.
    """
    stocknames = ['Tesla','Apple', 'Google', 'Amazon', 'Nvidia']

    # Create threads for each stock to speed up the scraping process
    threads = []
    for stockname in stocknames:
        thread = threading.Thread(target=scrape_and_insert_titles, args=(stockname, db))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Starting point for the script
if __name__ == "__main__":
    main()
