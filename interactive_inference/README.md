# Alpamayo 1.5 - Interactive Inference System

이 폴더는 무거운 Alpamayo 1.5 모델(10B)을 매번 로드하지 않고 유연하게 파라미터를 변경해 가며 추론(Inference)을 시각화하고 실험 결과를 관리하기 위해 만들어진 **웹 기반 인터랙티브 인퍼런스 시스템**입니다. 

RunPod 환경에서 작동하며, 실험 내역과 이미지는 모두 이 폴더 하위에 저장되므로 해당 폴더를 S3 등의 스토리지에 통째로 백업하시면 됩니다.

## 📁 폴더 구조

- `model_server.py` (포트 8000): 모델 로드와 추론을 담당하는 백엔드 서버
- `data_server.py` (포트 8002): 비디오 클립 데이터와 프레임 추출을 전담하는 데이터 서버
- `web_server.py` (포트 8888): 웹페이지(UI) 제공 및 SQLite DB 연동을 담당하는 프론트엔드 서버
- `templates/`: 화면 구성을 위한 HTML 파일들 (`index.html`, `data_check.html`, `inference.html`)
- `data/`: 사용자 실험 이력(SQLite), 생성된 차트 이미지, 그리고 추출된 프레임이 저장되는 장소

## 🚀 실행 방법

만약 서버 인스턴스를 껐다가 켰다면 아래의 순서대로 터미널에서 서버를 실행해주시면 됩니다.

### 1. 내부 모델 서버 켜기 (포트 8000)
가상 환경을 활성화하고 `model_server.py`를 실행합니다.
```bash
cd /workspace/Projects/alpamayo1.5
source a1_5_venv/bin/activate
python interactive_inference/model_server.py
```
> **참고**: 이 서버는 무거운 모델을 메모리에 로드하기 때문에 "Model loaded successfully!" 메시지가 뜰 때까지 약 30초~1분 정도 소요될 수 있습니다.

### 2. 데이터 전담 서버 켜기 (포트 8002)
새로운 터미널 탭에서 `data_server.py`를 실행합니다. 이 서버는 웹의 Data Check 메뉴에서 사진 슬라이더를 만드는 데 사용됩니다.
```bash
cd /workspace/Projects/alpamayo1.5
source a1_5_venv/bin/activate
python interactive_inference/data_server.py
```

### 3. 웹 프론트엔드 서버 켜기 (포트 8888)
새로운 터미널 창을 열고 `web_server.py`를 실행합니다.
```bash
cd /workspace/Projects/alpamayo1.5
source a1_5_venv/bin/activate
python interactive_inference/web_server.py
```
> **참고**: Jupyter Notebook이 8888 포트를 사용하고 있다면 포트 충돌이 날 수 있습니다. 이 경우 기존 Jupyter Notebook 프로세스를 종료(`pkill -f jupyter`)한 뒤 실행해주세요.

## 🌐 접속 및 사용 방법

1. **접속**: RunPod의 **Connect** 메뉴에서 제공하는 **[Port 8888 -> Jupyter notebook]** 링크를 클릭하면 브라우저에 웹 서버 UI가 열립니다.
2. **Data Check (데이터 확인)**
   - `clip_id`, `t0_relative`, 그리고 보고 싶은 **재생 시간(`duration_s`)**을 지정합니다.
   - 요청하면 데이터 서버에서 해당 구간 동안의 4채널 카메라 이미지들을 추출해옵니다. 
   - 화면 하단의 **슬라이더(재생바)**를 조절하여 시간별 상황을 매끄럽게 넘겨가며 확인할 수 있습니다!
3. **Interactive Inference (추론 실행 및 시각화)**
   - `nav_text` (예: "Turn right in 30m")와 각종 모델 파라미터(Temperature, Top-p 등)를 설정합니다.
   - `Runs (n)` 항목에 이 조건으로 추론을 몇 번 반복할지 입력합니다.
   - **Run Inference** 버튼을 누르면 내부적으로 모델 서버에 요청을 보내고, 처리된 결과물(Chain of Causation 텍스트, 각 케이스별 궤적 플롯 및 n번의 실행 평균 플롯)이 그려진 이미지를 확인하실 수 있습니다.
   - 모든 실행 내역과 설정된 파라미터는 `data/experiments.db`에 영구적으로 보존됩니다.

## 💾 백업 안내
이 시스템의 모든 데이터(이미지, 데이터베이스 기록)는 `data/` 폴더 안에 저장되며, 패키지 및 가상환경은 `a1_5_venv`에 구성되어 있습니다.
작업 완료 후 `/workspace` (혹은 `interactive_inference` 폴더)를 S3에 동기화하시면 모든 환경과 결과물이 안전하게 백업됩니다.
