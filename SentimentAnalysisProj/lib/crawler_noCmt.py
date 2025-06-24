import time
import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = 'https://www.inven.co.kr/board/maple'

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
})
REQUEST_TIMEOUT = 10

def make_driver():
    chrome_opts = Options()
    chrome_opts.add_argument('--headless')
    chrome_opts.add_argument('--disable-gpu')
    chrome_opts.add_argument('--no-sandbox')
    chrome_opts.page_load_strategy = 'eager'
    driver = webdriver.Chrome(options=chrome_opts)
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(10)
    return driver

driver = make_driver()

def fetch_posts(board_id, page=1, my_filter=None):
    params = {'p': page}
    if my_filter:
        params['my'] = my_filter
    res = session.get(f'{BASE_URL}/{board_id}', params=params, timeout=REQUEST_TIMEOUT)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'html.parser')
    posts = []
    rows = soup.select('form[name="board_list1"] .board-list table tbody tr')
    for row in rows:
        try:
            no     = row.select_one('td.num span').get_text(strip=True)
            a_tag  = row.select_one('td.tit a.subject-link')
            title  = a_tag.get_text(strip=True)
            link   = a_tag['href']
        except Exception:
            continue
        posts.append({
            '번호':   no,
            '제목':   title,
            '링크':   link,
        })
    return posts

def fetch_content_only(post_url):
    global driver
    for attempt in range(3):
        try:
            # 크롬드라이버 연결 상태 확인: 간단하게 현재 url 접근 시도
            try:
                _ = driver.title  # 강제 예외 유도
            except Exception:
                driver = make_driver()
            driver.get(post_url)
            break
        except (TimeoutException, WebDriverException) as e:
            try:
                driver.quit()
            except Exception:
                pass
            driver = make_driver()
            if attempt == 2:
                print(f"[ERROR] 본문 로드 최종 실패: {post_url}")
                return ''
            time.sleep(2)
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div#powerbbsContent'))
        )
    except TimeoutException:
        print(f"[WARN] 본문 요소 로드 타임아웃: {post_url}")
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    content_div = soup.find('div', id='powerbbsContent')
    content = content_div.get_text(strip=True) if content_div else ''
    time.sleep(2)
    return content
