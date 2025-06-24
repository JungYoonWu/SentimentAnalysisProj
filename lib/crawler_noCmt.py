# 파일명: lib/crawler_noCmt.py
import time
import requests
from bs4 import BeautifulSoup
import os
import sys

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
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
    """
    셀레니움 크롬 드라이버를 생성하고, 모든 종류의 불필요한 로그를 원천 차단합니다.
    """
    chrome_opts = Options()
    # --- ★ 로그 메시지 완전 차단을 위한 최종 옵션 ★ ---
    # 1. 기본적인 브라우저 옵션 설정
    chrome_opts.add_argument('--headless')          # 브라우저 창을 띄우지 않음
    chrome_opts.add_argument('--no-sandbox')        # 샌드박스 모드 비활성화
    chrome_opts.add_argument('--disable-dev-shm-usage') # /dev/shm 파티션 사용 비활성화
    chrome_opts.add_argument("--mute-audio")        # 음소거
    
    # 2. GPU 및 렌더링 관련 로그 차단 옵션
    chrome_opts.add_argument('--disable-gpu')       # GPU 가속 비활성화
    chrome_opts.add_argument('--disable-software-rasterizer') # 소프트웨어 래스터라이저 비활성화
    chrome_opts.add_argument('--disable-gpu-sandbox') # GPU 샌드박스 비활성화

    # 3. 터미널에 표시되는 로그 레벨 설정
    chrome_opts.add_argument('--log-level=3')       # 심각한 오류만 표시
    
    # 4. 실험적인 옵션을 통해 자동화 제어 메시지 및 로깅 기능 비활성화
    chrome_opts.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
    
    # 5. ChromeDriver 서비스 자체의 로그를 원천적으로 차단 (가장 중요)
    # 로그 파일 경로를 os.devnull (Windows에서는 'NUL')로 지정하여 모든 출력을 버림
    log_path = os.devnull if os.name != 'nt' else 'NUL'
    service_args = [
        '--log-level=OFF'  # 서비스 로그를 완전히 끔
    ]
    
    chrome_opts.page_load_strategy = 'eager'
    
    try:
        # WebDriverManager를 사용하여 드라이버를 자동 관리하고, 로그 경로와 인자를 전달
        service = Service(ChromeDriverManager().install(), log_path=log_path, service_args=service_args)
        driver = webdriver.Chrome(service=service, options=chrome_opts)
    except Exception as e:
        print(f"[드라이버 생성 오류] webdriver-manager 방식 실패, 기본 방식으로 재시도: {e}")
        # 실패 시 기본 방식도 동일한 서비스 인자를 사용하도록 시도
        service = Service(service_args=service_args, log_path=log_path)
        driver = webdriver.Chrome(service=service, options=chrome_opts)

    driver.set_page_load_timeout(60)
    driver.implicitly_wait(10)
    return driver

# 스크립트 시작 시 드라이버를 한 번만 생성
driver = make_driver()

def fetch_posts(board_id, page=1, my_filter=None):
    """게시판 목록 페이지에서 게시글 목록을 가져옵니다."""
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
            no_element = row.select_one('td.num span')
            a_tag = row.select_one('td.tit a.subject-link')
            if not no_element or not a_tag: continue
            no = no_element.get_text(strip=True)
            title = a_tag.get_text(strip=True)
            link = a_tag['href']
        except Exception:
            continue
            
        posts.append({
            '번호': no,
            '제목': title,
            '링크': link,
        })
    return posts

def fetch_content_only(post_url):
    """게시글 상세 페이지에서 본문 내용만 추출합니다."""
    global driver
    for attempt in range(3):
        try:
            try:
                _ = driver.title
            except WebDriverException:
                print("[INFO] 웹 드라이버를 재시작합니다...")
                driver.quit() # 기존 드라이버 확실히 종료
                driver = make_driver()

            driver.get(post_url)
            break
        except (TimeoutException, WebDriverException) as e:
            print(f"[WARN] 페이지 로드 실패 (시도 {attempt + 1}/3): {e}")
            try: driver.quit()
            except Exception: pass
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
    
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    content_div = soup.find('div', id='powerbbsContent')
    content = content_div.get_text(strip=True) if content_div else ''
    
    return content

# 스크립트 종료 시 드라이버 정리
import atexit
atexit.register(lambda: driver.quit() if driver else None)
