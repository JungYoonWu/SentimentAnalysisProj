# 파일명: crawler_main.py
import os
import csv
import pandas as pd
import time
from lib.crawler_noCmt import fetch_posts, fetch_content_only # lib 폴더는 그대로 유지

# 설정값들은 파일 상단에 정의하여 관리 용이성을 높입니다.
BOARDS = {
    '전사':     (2294, None),
    '마법사':   (2295, None),
    '궁수':     (2296, None),
    '도적':     (2297, None),
    '해적':     (2298, None),
    '자유':     (5974, None),
    '자유_30추': (5974, 'chuchu'),
    '자유_10추': (5974, 'chu'),
}
POSTS_CSV = './data/posts.csv'

def load_collected_post_ids():
    if not os.path.exists(POSTS_CSV):
        return set()
    try:
        df = pd.read_csv(POSTS_CSV, dtype=str)
        # '번호' 컬럼이 없는 경우를 대비
        if '번호' in df.columns:
            return set(df['번호'].astype(str))
        else:
            return set()
    except Exception:
        return set()

# 기존 crawl_all 함수를 파라미터를 받는 함수로 변경
def run_crawling(max_pages_per_board=5):
    """
    메이플 인벤 게시판을 크롤링하여 CSV 파일로 저장하는 메인 함수
    :param max_pages_per_board: 각 게시판에서 한 번에 크롤링할 최대 페이지 수
    """
    os.makedirs('./data', exist_ok=True)
    post_fields = ['게시판', '번호', '제목', '링크', '본문']
    collected_post_ids = load_collected_post_ids()
    post_file_exists = os.path.exists(POSTS_CSV)

    current_pages = {board: 1 for board in BOARDS.keys()}
    finished = set()

    with open(POSTS_CSV, 'a', newline='', encoding='utf-8-sig') as pf:
        pw = csv.DictWriter(pf, fieldnames=post_fields)
        if not post_file_exists or pf.tell() == 0:
            pw.writeheader()

        # 기존 로직은 거의 그대로 사용
        while len(finished) < len(BOARDS):
            for board_name, (bid, flt) in BOARDS.items():
                if board_name in finished:
                    continue

                page_start = current_pages[board_name]
                page_end = page_start + max_pages_per_board - 1
                empty_page_hit = False

                for page in range(page_start, page_end + 1):
                    posts = fetch_posts(bid, page, flt)
                    if not posts:
                        finished.add(board_name)
                        empty_page_hit = True
                        break
                    
                    print(f'[{board_name}] 페이지 {page} → {len(posts)}건')
                    for post in posts:
                        if post['번호'] in collected_post_ids:
                            continue
                        
                        try:
                            content = fetch_content_only(post['링크'])
                        except Exception as e:
                            print(f"[ERROR] fetch_content_only 실패: {post['링크']} → {e}")
                            content = ''
                        
                        pw.writerow({'게시판': board_name, '번호': post['번호'], '제목': post['제목'], '링크': post['링크'], '본문': content})
                        collected_post_ids.add(post['번호'])
                        time.sleep(2) # 서버 부하 방지를 위한 대기 시간
                
                if not empty_page_hit:
                    current_pages[board_name] += max_pages_per_board

    print(f'크롤링 완료 → {POSTS_CSV}')

# 이 스크립트 파일을 직접 실행할 때만 아래 코드가 동작
if __name__ == '__main__':
    # 테스트 실행: 각 게시판별로 5페이지씩 크롤링
    run_crawling(max_pages_per_board=5)