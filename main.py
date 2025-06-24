# 파일명: main.py
import os
import json
import pandas as pd
import subprocess
import sys
import glob
import datetime
import shutil
import time
import multiprocessing

# --- 설정 ---
ALL_POSTS_CSV = './data/all_posts.csv'
RAW_CHUNK_DIR = './data/raw_chunks/'
LABELED_CHUNK_DIR = './data/labeled_chunks/'
FEEDBACK_FILE_PATH = './data/user_feedback.csv'
ARCHIVED_FEEDBACK_DIR = './data/archived_feedback/'
STATE_FILE = './data/pipeline_state.json'
CHUNK_SIZE = 1000
FEEDBACK_TRIGGER_COUNT = 100
ORCHESTRATOR_SLEEP_MINUTES = 5

# --- 가상환경 경로 ---
CRAWLER_PYTHON_PATH = r"C:\Users\User\anaconda3\envs\nlp-tfgpu\python.exe"
LABELER_PYTHON_PATH = r"C:\Users\User\anaconda3\envs\LangchainEnv\python.exe"
TRAINING_PYTHON_PATH = r"C:\Users\User\anaconda3\envs\nlp-tfgpu\python.exe"

def run_script(python_path, *args):
    """지정된 파이썬 인터프리터로 스크립트를 실행하고, 인코딩 문제를 해결합니다."""
    script_name = args[0]
    print(f"\n--- '{script_name}' 실행 (Python: {python_path}) ---")
    
    try:
        # --- ★ 한글 깨짐 방지를 위한 환경 변수 설정 ★ ---
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        process = subprocess.Popen(
            [python_path, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env, # ★ 수정된 환경 변수 전달
            encoding='utf-8', 
            errors='replace'
        )
        
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[오류] '{script_name}' 실행 중 오류 발생. 종료 코드: {process.returncode}")
            return False
        
        print(f"--- '{script_name}' 성공적으로 완료 ---")
        return True
        
    except FileNotFoundError:
        print(f"\n[오류] 파이썬 실행 파일을 찾을 수 없습니다: '{python_path}'")
        return False
    except Exception as e:
        print(f"\n[오류] '{script_name}' 실행 중 예외 발생: {e}")
        return False

# (이하 get_state, save_state, crawler_worker, orchestrator_worker 등의 함수는 이전과 동일)
def get_state():
    if not os.path.exists(STATE_FILE): return {'processed_rows': 0, 'last_chunk_number': 0}
    try:
        with open(STATE_FILE, 'r') as f: return json.load(f)
    except: return {'processed_rows': 0, 'last_chunk_number': 0}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=4)

def archive_feedback_file():
    if not os.path.exists(FEEDBACK_FILE_PATH): return
    os.makedirs(ARCHIVED_FEEDBACK_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(ARCHIVED_FEEDBACK_DIR, f'feedback_{timestamp}.csv')
    shutil.move(FEEDBACK_FILE_PATH, archive_path)
    print(f"피드백 파일이 '{archive_path}'로 아카이브되었습니다.")

def find_latest_labeled_chunk():
    if not os.path.exists(LABELED_CHUNK_DIR): return None
    labeled_files = glob.glob(os.path.join(LABELED_CHUNK_DIR, 'labeled_data_*.csv'))
    return max(labeled_files, key=os.path.getctime) if labeled_files else None

def crawler_worker():
    print("[프로세스 1: 크롤러] 시작")
    run_script(CRAWLER_PYTHON_PATH, 'crawler_main.py')

def orchestrator_worker():
    print("[프로세스 2: 오케스트레이터] 시작")
    while True:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] [오케스트레이터] 새로운 데이터가 있는지 확인합니다...")
        path_for_training = None
        feedback_triggered = False
        state = get_state()
        processed_rows = state.get('processed_rows', 0)
        try:
            if os.path.exists(ALL_POSTS_CSV):
                all_df = pd.read_csv(ALL_POSTS_CSV)
                total_rows = len(all_df)
                if total_rows >= processed_rows + CHUNK_SIZE:
                    print(f"[오케스트레이터] 신규 데이터 {total_rows - processed_rows}건 발견! 처리를 시작합니다.")
                    new_chunk_df = all_df.iloc[processed_rows : processed_rows + CHUNK_SIZE]
                    chunk_num = state.get('last_chunk_number', 0) + 1
                    raw_chunk_path = os.path.join(RAW_CHUNK_DIR, f'raw_data_{chunk_num:02}.csv')
                    labeled_chunk_path = os.path.join(LABELED_CHUNK_DIR, f'labeled_data_{chunk_num:02}.csv')
                    new_chunk_df.to_csv(raw_chunk_path, index=False, encoding='utf-8-sig')
                    if run_script(LABELER_PYTHON_PATH, 'labeler_main.py', raw_chunk_path, labeled_chunk_path):
                        path_for_training = labeled_chunk_path
                        state['processed_rows'] = processed_rows + CHUNK_SIZE
                        state['last_chunk_number'] = chunk_num
                        save_state(state)
        except Exception as e:
            print(f"[오케스트레이터 오류] 데이터 처리 중 문제 발생: {e}")
        try:
            if os.path.exists(FEEDBACK_FILE_PATH):
                feedback_df = pd.read_csv(FEEDBACK_FILE_PATH)
                if len(feedback_df) >= FEEDBACK_TRIGGER_COUNT:
                    print(f"[오케스트레이터] 사용자 피드백 {len(feedback_df)}건 발견! 처리를 시작합니다.")
                    feedback_triggered = True
                    if not path_for_training:
                        path_for_training = find_latest_labeled_chunk()
        except Exception as e:
            print(f"[오케스트레이터 오류] 피드백 데이터 확인 중 문제 발생: {e}")
        if path_for_training:
            print("[오케스트레이터] 모델 재학습을 시작합니다...")
            if run_script(TRAINING_PYTHON_PATH, 'training_pipeline.py', path_for_training):
                if feedback_triggered:
                    archive_feedback_file()
        else:
            print("[오케스트레이터] 재학습 조건이 충족되지 않았습니다.")
        print(f"[오케스트레이터] 확인 완료. {ORCHESTRATOR_SLEEP_MINUTES}분 후 다시 확인합니다.")
        time.sleep(ORCHESTRATOR_SLEEP_MINUTES * 60)

if __name__ == '__main__':
    os.makedirs(RAW_CHUNK_DIR, exist_ok=True)
    os.makedirs(LABELED_CHUNK_DIR, exist_ok=True)
    os.makedirs(ARCHIVED_FEEDBACK_DIR, exist_ok=True)

    print("="*80)
    print("병렬 처리 MLOps 파이프라인을 시작합니다.")
    print("프로세스 1 (크롤러)와 프로세스 2 (오케스트레이터)를 동시에 실행합니다.")
    print("(Ctrl+C로 두 프로세스를 모두 종료)")
    print("="*80)

    p_crawler = multiprocessing.Process(target=crawler_worker)
    p_crawler.start()
    p_orchestrator = multiprocessing.Process(target=orchestrator_worker)
    p_orchestrator.start()
    
    try:
        p_crawler.join()
        p_orchestrator.join()
    except KeyboardInterrupt:
        print("\n[메인 프로세스] 종료 신호(Ctrl+C) 수신. 자식 프로세스를 종료합니다.")
        p_crawler.terminate()
        p_orchestrator.terminate()
        p_crawler.join()
        p_orchestrator.join()
        print("[메인 프로세스] 모든 프로세스가 종료되었습니다.")
