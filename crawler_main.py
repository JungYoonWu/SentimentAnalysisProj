# 파일명: crawler_main.py
import os
import csv
import pandas as pd
import time
import sys
import glob

# 스크립트의 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.crawler_noCmt import fetch_posts, fetch_content_only

# --- 설정 ---
ALL_POSTS_CSV = './data/all_posts.csv'
BOARDS = {
    '전사': (2294, None), '마법사': (2295, None), '궁수': (2296, None),
    '도적': (2297, None), '해적': (2298, None), '자유': (5974, None),
    '자유_30추': (5974, 'chuchu'), '자유_10추': (5974, 'chu'),
}
# 각 게시판에서 최대 몇 페이지까지 확인할지 설정
MAX_PAGES_PER_BOARD = 5
# 전체 크롤링 사이클 후 몇 분 대기할지 설정 (초 단위)
SLEEP_TIME_SECONDS = 60 * 10 # 10분

def get_all_collected_ids():
    """모든 기존 데이터에서 수집된 게시글 ID를 전부 가져옵니다."""
    if not os.path.exists(ALL_POSTS_CSV):
        return set()
    try:
        df = pd.read_csv(ALL_POSTS_CSV, dtype={'번호': str})
        return set(df['번호'].tolist())
    except Exception:
        return set()

def run_continuous_crawling():
    """
    무한 루프를 돌면서 주기적으로 새로운 게시글을 수집합니다.
    """
    print("[크롤러 시작] 멈추지 않는 데이터 수집을 시작합니다...")
    os.makedirs('./data', exist_ok=True)
    post_fields = ['게시판', '번호', '제목', '본문', '링크']
    
    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 새로운 크롤링 사이클 시작...")
        collected_post_ids = get_all_collected_ids()
        print(f"현재까지 수집된 총 ID 수: {len(collected_post_ids)}")
        
        file_exists = os.path.exists(ALL_POSTS_CSV)
        new_posts_in_cycle = 0

        with open(ALL_POSTS_CSV, 'a', newline='', encoding='utf-8-sig') as pf:
            pw = csv.DictWriter(pf, fieldnames=post_fields)
            if not file_exists or pf.tell() == 0:
                pw.writeheader()

            for board_name, (bid, flt) in BOARDS.items():
                print(f"--- '{board_name}' 게시판 확인 중...")
                for page in range(1, MAX_PAGES_PER_BOARD + 1):
                    try:
                        posts = fetch_posts(bid, page, flt)
                        if not posts: break
                        
                        new_in_page = 0
                        for post in posts:
                            if post['번호'] not in collected_post_ids:
                                content = fetch_content_only(post['링크'])
                                pw.writerow({
                                    '게시판': board_name, '번호': post['번호'],
                                    '제목': post['제목'], '본문': content, '링크': post['링크']
                                })
                                collected_post_ids.add(post['번호'])
                                new_posts_in_cycle += 1
                                new_in_page += 1
                                print(f"    -> 신규 게시글 수집 : {post['제목'][:30]}")
                                time.sleep(1.5)
                        
                        if new_in_page == 0:
                            # 현재 페이지에 새 글이 없으면 다음 게시판으로
                            break

                    except Exception as e:
                        print(f"[오류] '{board_name}' 게시판 페이지 {page} 처리 중: {e}")
                        time.sleep(5)
                        continue
        
        print(f"\n이번 사이클에서 총 {new_posts_in_cycle}개의 새로운 데이터를 수집했습니다.")
        print(f"다음 크롤링 사이클까지 {SLEEP_TIME_SECONDS / 60:.0f}분 대기합니다...")
        time.sleep(SLEEP_TIME_SECONDS)

if __name__ == '__main__':
    run_continuous_crawling()
