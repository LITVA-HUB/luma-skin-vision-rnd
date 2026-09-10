# Основные внешние источники
Проверка: 10 сентября 2026 года. Ссылки нужны для воспроизводимого изучения, а не являются разрешением использовать чужие модели, данные или код. Лицензии code / weights / dataset проверять отдельно, фиксируя revision и SHA.

## Близкий prior art
- TRUST (ECCV 2022): https://arxiv.org/abs/2205.03962 ; https://github.com/HavenFeng/TRUST . Код и модель — non-commercial scientific research, не production-компонент без отдельного разрешения.
- C5 (ICCV 2021): https://arxiv.org/abs/2011.11890 ; https://github.com/mahmoudnafifi/C5 . Код Apache-2.0; RAW / данные новой камеры; лицензии datasets проверять отдельно.
- Deep White Balance (CVPR 2020): https://arxiv.org/abs/2004.01354 ; https://github.com/mahmoudnafifi/Deep_White_Balance . Опубликованный код research-only; не считать коммерчески свободным.
- Region-Specific Calibration, v4 от 2 июля 2026: https://arxiv.org/abs/2512.21988v4 ; https://arxiv.org/html/2512.21988v4 . Важно: ранняя версия имела другое название и выводы; не смешивать версии. DSLR-референс не спектрофотометрический эталон. Код заявлен MIT: https://github.com/hpicsk/regional-ccm ; данные AI Hub с отдельным доступом/условиями.
- True to Tone? (апрель 2026): https://arxiv.org/abs/2604.02055 . Оценка фото→виртуальный человек, TRUST и выбор зон; не benchmark точности цвета обычного смартфона относительно прибора.
- Colorimeter-Supervised Skin Tone Estimation (февраль 2026): https://arxiv.org/abs/2602.10265 ; https://arxiv.org/html/2602.10265v1 . Дерматоскопия, не обычные фотографии лица. Код: https://github.com/marinbenc/nn_colorimetry_dermatoscopy ; коммерческая цепочка прав не подтверждена.
- Smartphone tristimulus colorimetry (2024): https://arxiv.org/abs/2411.13832 . Четыре участника, не популяционная валидация на неизвестных телефонах.
- Colorimetric skin tone scale (2024): https://arxiv.org/abs/2410.21005 . Измерительный и аннотационный prior art.
- SREDS generalizability (2023): https://arxiv.org/abs/2309.01235 . Дихроматическая модель цвета.
- Synthetic dermatoscopy skin-color evaluation (2025): https://arxiv.org/abs/2504.04494 . Синтетика не заменяет реальные приборные измерения лица.
- Generative Color Constancy (2025): https://arxiv.org/abs/2502.17435 . Синтетический ColorChecker — модельная оценка, не измерительный эталон.
- SelectiveNet (2019): https://arxiv.org/abs/1901.09192 . Prediction + reject head уже известны.
- Calibration (Guo et al., 2017): https://proceedings.mlr.press/v70/guo17a.html . Не переносить temperature scaling классификатора автоматически на надежность регрессии цвета.
- Learn then Test: https://arxiv.org/abs/2110.01052 . Контроль риска при выборе параметров; предпосылки проверять.
- Conformal Risk Control: https://arxiv.org/abs/2208.02814 . Не обещает универсальную гарантию при произвольном domain shift.
- Uncertainty under shift: https://arxiv.org/abs/1906.02530 . Проверять калибровку отдельно на camera/light shift.

## Компактные модели и лицензии
- YuNet: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet . MIT в каталоге модели, включая опубликованные веса; точная версия Luma указана в audit_evidence.md.
- MobileNetV4 paper: https://arxiv.org/abs/2404.10518 ; выбранная model card: https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k . Apache-2.0 на карточке; около 3.77M параметров у классификатора, новая конфигурация с decoder еще не измерена.
- MobileOne: https://github.com/apple/ml-mobileone ; https://raw.githubusercontent.com/apple/ml-mobileone/main/LICENSE . Собственная permissive-лицензия Apple, НЕ называть MIT; без patent grant.
- FastViT: https://github.com/apple/ml-fastvit ; https://raw.githubusercontent.com/apple/ml-fastvit/main/LICENSE . Собственная лицензия Apple; не путать с MIT.
- DINOv2: https://github.com/facebookresearch/dinov2 . Generic DINOv2 code/weights Apache-2.0; специальные XRay/Cell варианты имеют другие, в том числе NC, условия.
- Face parsing: https://github.com/zllrunning/face-parsing.PyTorch . MIT-код не очищает права на веса, обученные на CelebAMask-HQ.
- CelebAMask-HQ: https://github.com/switchablenorms/CelebAMask-HQ . Non-commercial research; не брать как коммерческий dataset без разрешения.
- SegFormer upstream license: https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE . Non-commercial research/evaluation, не blanket Apache.
- MediaPipe: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker . Лицензия страницы/примеров не является автоматически лицензией task-весов; отдельный gate перед использованием.
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- TensorRT support matrix: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- CIE Colorimetry: https://cie.co.at/publications/colorimetry-4th-edition

## Коммерческие аналоги и патентный ориентир
- Perfect Corp AI Shade Finder: https://www.perfectcorp.com/business/products/ai-shade-finder . Поставщик уже описывает контроль освещения, позиции и угла. Публичного сопоставимого ΔE00/risk–coverage benchmark в проверенной странице нет.
- ModiFace: https://modiface.com/skin-analysis . Закрытая технология, не open-source baseline.
- US20230074782A1: https://patents.google.com/patent/US20230074782A1/en . Заявка на сканирование и подбор косметики; в независимом п.1 есть контактное ручное устройство. Наличие публикации не доказывает запрет обычного smartphone pipeline; нужен разбор формулы, семейства, стран и действующего статуса патентным специалистом.

## Официальные материалы Сколково и налоговые нормы
- Критерии и процесс подачи: https://sk.ru/applicants-actions/
- Приоритеты, страница указывает редакцию 10.04.2026 №99-Пр: https://sk.ru/foundation/innovacionnye-prioritety/ . Содержимое динамического перечня/точный код для Luma не удалось подтвердить; уточнить перед подачей.
- Правила исследовательской деятельности: текущая ссылка с официальной страницы подачи https://sk.ru/applicants-actions/ (раздел «правила проекта»). Прочитана опубликованная редакция декабря 2025 года, включая страницы 1/4/6 PDF визуально.
- ФНС: https://www.nalog.gov.ru/rn77/taxation/taxes/profitul/
- НК через правовой раздел ФНС, ст.262: https://nalog.garant.ru/fns/nk/461305094b1a78956a2b088608d7ba0c/
- Ст.427: https://nalog.garant.ru/fns/nk/3c9c72380388b707a88dcf14d96be986/ . Отдельно проверить ограничения совместного использования ИТ-тарифов и статуса Сколково.
- Ст.149: https://nalog.garant.ru/fns/nk/11e2106fa4ec328ea2d88df540010b52/ . Реестр ПО не означает освобождение любой marketplace/recommendation операции.
- Ст.145.1: https://nalog.garant.ru/fns/nk/f64a2020c0eadcbde4309b4b740aae87/
- Ст.246.1: https://nalog.garant.ru/fns/nk/d3e00504a92344f4cf207fa82c896bd2/

## Нормативный текст интеллектуальных прав
Тексты ГК использованы в правовой системе КонсультантПлюс, не SEO-пересказы:
- 1465: https://www.consultant.ru/document/cons_doc_LAW_64629/f8743d677137889f4c521c9e3e17a5d837ed54bf/
- 1467: https://www.consultant.ru/document/cons_doc_LAW_64629/d38291d12acca233c5000a3ae563f82c1e0cac14/
- 1262: https://www.consultant.ru/document/cons_doc_LAW_64629/d0887a7ca3da6c85fbbce19815b9b1ead5e67687/
- 1350: https://www.consultant.ru/document/cons_doc_LAW_64629/4b30fa7ca4e5733597a1bc9b2b12351cc5c430e6/ . Не смешивать действующий текст 2026 года и отмеченные изменения с 01.01.2027.

## Пробелы проверки
Полные актуальные официальные регламенты ИТ-аккредитации и включения в реестр российского ПО не были доступны через проверенные веб-страницы. Не считать этот пакет юридическим заключением или полным чек-листом Минцифры. Действующие редакции и применимость конкретному юрлицу должен подтвердить профильный специалист. Статус Luma, структура собственников, доходы и оформленные права документами архива не подтверждены.
