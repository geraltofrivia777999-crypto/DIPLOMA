# AntiTerror — интеграция с фронтендом

Этот проект предоставляет Python‑слой интеграции, который позволяет фронтенду
получать несколько потоков камер с JPEG‑кадрами и живой статистикой.

## Запуск проекта

### 1. Подготовка окружения

Windows / Linux:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-api.txt
```

Важно:
- используйте `numpy<2` из `requirements.txt`; если в окружении уже установлен `numpy 2.x`, переустановите зависимости;
- для запуска на NVIDIA GPU нужен CUDA-совместимый `torch` и рабочие библиотеки cuDNN;
- если GPU не нужен или CUDA ещё не настроена, запускайте с `--device cpu`.

### 2. Быстрый локальный запуск без PostgreSQL

Этот режим подходит для проверки камеры, детекции, трекинга и распознавания лиц без API-дашборда.

1. Убедитесь, что запуск идёт в SQLite-режиме:

```powershell
python main.py --source 0 --device cpu --db-backend sqlite --db antiterror.db
```

Если CUDA уже настроена:

```powershell
python main.py --source 0 --device cuda --db-backend sqlite --db antiterror.db
```

Примеры:
- веб-камера: `--source 0`
- видеофайл: `--source "video.mp4"`
- RTSP: `--source "rtsp://user:pass@host/stream"`

Если нужен просмотр в браузере без OpenCV-окна:

```powershell
python main.py --source 0 --device cpu --db-backend sqlite --db antiterror.db --preview-port 8081
```

После этого MJPEG-поток будет доступен по адресу:

```text
http://127.0.0.1:8081/
```

Что проверять:
- приложение не падает на первом кадре;
- в логах появляется строка `Face embedder ready (...)`;
- на кадре рисуются рамки лиц и идентификаторы `P_0001`, `P_0002`, ...;
- один и тот же человек не получает новый ID каждые несколько кадров.

### 3. Запуск с PostgreSQL и API

Этот режим нужен для веб-дашборда, API-роутов и поиска по лицу через `pgvector`.

1. Проверьте `.env`:

```env
DB_BACKEND=postgres
POSTGRES_DSN=postgresql://antiterror:antiterror@localhost:5432/antiterror
POSTGRES_DB=antiterror
POSTGRES_USER=antiterror
POSTGRES_PASSWORD=antiterror
POSTGRES_PORT=5432
API_HOST=0.0.0.0
API_PORT=8000
```

2. Поднимите PostgreSQL из `docker-compose`:

```powershell
docker compose -f docker/docker-compose.yml up -d
```

Контейнер сам включает `pgvector`, а схема таблиц создаётся автоматически при первом подключении через `PostgresDatabase`.

3. Запустите пайплайн:

```powershell
python main.py --source 0 --device cpu --db-backend postgres
```

Или с CUDA:

```powershell
python main.py --source 0 --device cuda --db-backend postgres
```

4. Если нужен поток камеры в дашборде, поднимите пайплайн с preview-портом:

```powershell
python main.py --source 0 --device cpu --db-backend postgres --preview-port 8081
```

5. В отдельном терминале запустите API:

```powershell
python serve.py --host 127.0.0.1 --port 8000
```

6. Откройте в браузере:
- дашборд: `http://127.0.0.1:8000/`
- список камер: `http://127.0.0.1:8000/api/v1/cameras`
- статистика: `http://127.0.0.1:8000/api/v1/stats`
- MJPEG proxy потока: `http://127.0.0.1:8000/api/v1/cameras/CAM_01/stream`

### 4. Частые проблемы

- `RuntimeError: Numpy is not available`: в окружении установлен `numpy 2.x`, а часть бинарных пакетов собрана под `numpy 1.x`. Переустановите `numpy` на `<2`.
- `Error loading onnxruntime_providers_cuda.dll ... cudnn64_9.dll is missing`: `onnxruntime-gpu` не может использовать CUDA; InsightFace уйдёт на CPU, пока не будет установлен cuDNN 9.
- `Database backend: postgres`, но API не стартует: проверьте, что контейнер PostgreSQL поднят и `POSTGRES_DSN` доступен.
- API запущен, но камера в дашборде не отображается: пайплайн должен быть запущен с `--preview-port`, а в `anti_terror/api/settings.py` или `.env` должен совпадать `preview_streams` (по умолчанию `CAM_01=8081`).

## Быстрый старт (несколько камер)

```python
from anti_terror.service import MultiCameraService, CameraConfig

svc = MultiCameraService()

# Опционально: получать события/алерты
def on_events(camera_id, events):
    print("events", camera_id, events)

svc.set_event_callback(on_events)

# Запуск камер
svc.start_camera(CameraConfig(camera_id="CAM_01", source=0, db_path="antiterror.db"))
svc.start_camera(CameraConfig(camera_id="CAM_02", source="rtsp://user:pass@host/stream"))

# Получить последний JPEG/статистику (для передачи во фронтенд)
jpeg = svc.get_latest_jpeg("CAM_01")
stats = svc.get_latest_stats("CAM_01")

# Остановить камеры
svc.stop_camera("CAM_01")
svc.stop_all()
```

## API интеграции

### MultiCameraService

- `start_camera(cfg: CameraConfig)`: запускает пайплайн в фоне.
- `stop_camera(camera_id: str)`: останавливает одну камеру.
- `stop_all()`: останавливает все камеры.
- `get_latest_jpeg(camera_id: str) -> bytes | None`: последний кадр в JPEG.
- `get_latest_stats(camera_id: str) -> dict | None`: последняя статистика.
- `set_event_callback(callback)`: колбек получает `(camera_id, events)`.

### CameraConfig

```python
CameraConfig(
    camera_id: str,        # unique camera identifier
    source: str | int,     # webcam index or path/RTSP URL
    db_path: str = "antiterror.db",
    device: str = "cuda",  # "cuda" | "mps" | "cpu"
    preview_port: int | None = None,
    render_enabled: bool = False,
)
```

Примечания:
- `preview_port` запускает простой MJPEG‑сервер (для быстрых проверок).
- `render_enabled` включает OpenCV‑окно (полезно только локально).

## Форматы данных

### Статистика

```python
{
  "camera_id": "CAM_01",
  "faces": 3,
  "ids": 2,
  "bags": 1,
  "owned": 1,
  "session_id": "S_2026_02_03_0001"
}
```

### События

`events` — список словарей, формируемых анализатором поведения; каждый элемент
минимум содержит `type`, а также может включать `bag_id`, `person_id` и другие
поля в зависимости от типа события.

## Рекомендации по отдаче во фронтенд

Два распространённых варианта интеграции:

1) HTTP API:
   - Сделать небольшой API, который отдаёт `get_latest_jpeg` и `get_latest_stats`.
   - Фронтенд опрашивает API и рисует кадры как изображения.

2) WebSocket/Stream:
   - Пушить JPEG и статистику по WS для меньшей задержки.

Если нужно, добавлю FastAPI‑сервер с эндпоинтами:
- `/cameras` список
- `/camera/{id}/jpeg`
- `/camera/{id}/stats`
- `/events` поток

## Диагностика

- Если камера не открывается, проверьте корректность `source`.
- Если кадров нет, проверьте доступ к GPU или переключитесь на `device="cpu"`.
- RTSP‑потоки часто требуют корректных логина/пароля и сетевого доступа.
