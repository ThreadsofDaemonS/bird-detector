# Bird Detector

Детекція птахів на зображеннях і відео. Модель **YOLO26n** (Ultralytics), донавчена на одному класі — `bird`.

Результат роботи на прикладі (`examples/output.mp4`): рамка, назва класу та confidence на кожному кадрі.

## Структура проєкту

```
bird-detector/
├── README.md
├── requirements.txt
├── prepare_dataset.py   # підготовка датасету (COCO -> YOLO, train/val)
├── train.py             # навчання
├── predict_video.py     # інференс на відео
├── train_colab.ipynb    # те саме навчання в Google Colab (GPU)
├── weights/
│   └── best.pt          # навчені ваги
└── examples/
    ├── input.mp4
    └── output.mp4
```

## Встановлення

Потрібен Python 3.10+.

```bash
git clone https://github.com/ThreadsofDaemonS/bird-detector.git
cd bird-detector
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Швидка перевірка (готові ваги)

Ваги `weights/best.pt` вже є в репозиторії, тож навчати нічого не треба:

```bash
python predict_video.py --input examples/input.mp4 --output examples/output.mp4
```

Опції: `--weights` (за замовчуванням `weights/best.pt`), `--conf` (поріг впевненості, за замовчуванням 0.25), `--imgsz` (640), `--device` (`cpu` або `0` для GPU).

Скрипт читає відео покадрово, запускає модель на кожному кадрі, малює bounding box з підписом `bird <confidence>` і зберігає результат у mp4 (H.264).

## Навчання з нуля

1. Підготувати датасет (завантажує ~200 МБ, 2–3 хв):

```bash
python prepare_dataset.py --num-images 1200
```

2. Запустити навчання:

```bash
python train.py
```

За замовчуванням: `--epochs 30 --imgsz 640 --batch 16`. Після навчання скрипт перевіряє найкращий чекпойнт на validation, друкує метрики та копіює ваги в `weights/best.pt`.

На CPU одна епоха займає ~11 хв (Intel UHD, без CUDA), тож 30 епох — понад 5 годин. Я навчав на безкоштовному GPU T4 в Google Colab: відкрийте `train_colab.ipynb`, оберіть **T4 GPU** та виконайте всі клітинки (**50 епох**, `imgsz=640`, `batch=16`). Останній крок завантажує `best.pt`.

## Датасет

- **Джерело:** [COCO 2017](https://cocodataset.org/), клас `bird` (train2017). Ліцензія зображень — CC BY 2.5 (Flickr).
- **Вибірка:** 1200 випадкових зображень із птахами (`seed=0`), 3284 bounding box-и.
- **Розбиття:** 960 train / 240 val (80/20). На val — 637 птахів.
- **Розмітка:** COCO JSON переведено у формат YOLO (`0 cx cy w h`, координати нормалізовані), один клас `bird` (id 0). Відкинуто `iscrowd`-розмітку (групові рамки) і box-и менші за 0.05% площі кадру.
- Усе відтворюється командою `python prepare_dataset.py`.

## Метрики на validation

Ваги `weights/best.pt`, 240 зображень, 637 об'єктів, `imgsz=640`:

| Precision | Recall | mAP50 | mAP50-95 |
|-----------|--------|-------|----------|
| 0.702     | 0.604  | 0.656 | 0.399    |

Для порівняння: після 1 епохи mAP50 був 0.095.

## Як переконатися, що модель бачить птахів

1. Запустити `predict_video.py` (див. вище) і відкрити `examples/output.mp4`. На птахах мають бути рамки з підписом `bird 0.xx`.
2. Передивитися на довільному власному відео: `python predict_video.py --input my.mp4 --output out.mp4`.
3. Знизити `--conf 0.1`, якщо рамок мало: з'являться й менш впевнені детекції.
4. Для швидкої перевірки на одній картинці:
   ```bash
   yolo predict model=weights/best.pt source=path/to/image.jpg save=True
   ```
5. Перерахувати метрики: `yolo val model=weights/best.pt data=datasets/birds/data.yaml`.

## Проблеми та можливі покращення

**Що виявилось на практиці**

- **Дрібні птахи.** На відео зі зграєю чайок у небі (птахи кілька пікселів) модель не знайшла жодного птаха, хоча на відео з качками й голубями працює добре. Причини: nano-модель, `imgsz=640` і те, що дрібні рамки (<0.05% кадру) я відфільтрував із датасету, тож модель таких майже не бачила.
- **Птахи на краю кадру.** Частково обрізані птахи часто не детектуються.
- **Малий датасет.** 960 train-зображень замало; COCO зміщений до фото, де птах великий і по центру, тож на «реальних» відео (небо, далека відстань) якість нижча за метрики на val.
- **Навчання на CPU.** Без GPU навчання надто повільне, тому використано Colab.
- **mAP50-95 (0.40)** значно нижче за mAP50 (0.66): знаходить птахів, але рамки не завжди точні.
- Val-вибірка взята з того ж розподілу (COCO), тому метрики оптимістичніші за реальні умови.

**Можливі покращення**

- Більше даних (весь клас `bird` з COCO, Open Images, CUB-200, NABirds), різноманітніші сцени, зокрема птахи в польоті й на далекій відстані.
- Більша модель (`yolo26s/m`) і довше навчання.
- Вищий `imgsz` (960–1280) або інференс тайлами (SAHI) для дрібних об'єктів.
- Аугментації для дрібних об'єктів (mosaic, scale, copy-paste).
- Трекінг (ByteTrack: `model.track`) для стабільних рамок між кадрами й підрахунку птахів.
- Окремий тестовий набір відео, а не лише val з COCO.

## Ліцензії та атрибуція

- Датасет: COCO 2017, зображення — CC BY 2.5 (власники — автори з Flickr).
- Приклад відео: «Mallards pond 2009», Wikimedia Commons, ліцензія [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). Відео обрізано та перекодовано.
- Модель і бібліотека: [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
