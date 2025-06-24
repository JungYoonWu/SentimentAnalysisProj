# 파일명: main.py
import subprocess
import sys
from apscheduler.schedulers.blocking import BlockingScheduler

# --- 각 작업에 맞는 가상환경의 Python 실행 파일 경로 ---
CRAWLER_PYTHON_PATH = r"C:\Users\User\anaconda3\envs\nlp-tfgpu\python.exe"
LABELER_PYTHON_PATH = r"C:\Users\User\anaconda3\envs\LangchainEnv\python.exe"
# 모델 학습도 nlp-tfgpu 환경을 사용하므로 경로를 미리 정의해둡니다.
TRAINING_PYTHON_PATH = r"C:\Users\User\anaconda3\envs\nlp-tfgpu\python.exe"

def run_script(python_path, script_name):
    """지정된 파이썬 인터프리터로 스크립트를 실행하는 함수"""
    try:
        process = subprocess.Popen(
            [python_path, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace' # 인코딩 에러 발생 시 글자 깨짐 방지
        )
        
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[오류] '{script_name}' 실행 중 오류 발생. 종료 코드: {process.returncode}")
            return False
        return True
    except FileNotFoundError:
        print(f"\n[오류] 파이썬 실행 파일을 찾을 수 없습니다: '{python_path}'")
        return False
    except Exception as e:
        print(f"\n[오류] 스크립트 실행 중 예외 발생: {e}")
        return False

def maple_sentiment_pipeline_job():
    """데이터 수집부터 모델 학습/배포까지 이어지는 전체 파이프라인 잡"""
    print("="*80)
    print(f"자동화된 ML 파이프라인을 시작합니다.")
    
    # 1. 크롤링 단계 (nlp-tfgpu 환경으로 실행)
    print("\n[1/3] 크롤링을 시작합니다...")
    if not run_script(CRAWLER_PYTHON_PATH, "crawler_main.py"):
        print("크롤링 단계 실패. 파이프라인을 중단합니다.")
        return
    print("크롤링 단계 성공적으로 완료.")

    # 2. 라벨링 단계 (LangchainEnv 환경으로 실행)
    print("\n[2/3] LLM 라벨링을 시작합니다...")
    if not run_script(LABELER_PYTHON_PATH, "labeler_main.py"):
        print("라벨링 단계 실패. 파이프라인을 중단합니다.")
        return
    print("라벨링 단계 완료.")

    # 3. 학습/평가/배포 단계 (다음 코드 공유 후 채울 예정)
    print("\n[3/3] 모델 학습 및 평가/배포를 시작합니다...")
    # if not run_script(TRAINING_PYTHON_PATH, "training_pipeline.py"):
    #     print("학습/평가 단계 실패. 파이프라인을 중단합니다.")
    #     return
    print("학습/평가 파이프라인은 아직 구현되지 않았습니다.")
    
    print("\n모든 단계가 완료되었습니다.")
    print("="*80)

# --- 스케줄러 설정 ---
scheduler = BlockingScheduler(timezone='Asia/Seoul')
scheduler.add_job(maple_sentiment_pipeline_job, 'cron', hour=4, minute=0, id='maple_pipeline')

print("스케줄러가 시작되었습니다. 매일 정해진 시간에 파이프라인이 실행됩니다.")
print("첫 작업은 설정된 가장 빠른 시간에 실행됩니다. (Ctrl+C로 종료)")

try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
