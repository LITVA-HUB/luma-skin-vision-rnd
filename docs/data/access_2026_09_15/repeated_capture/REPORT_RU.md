| dataset | downloaded | people | images | repeated captures | camera/illumination variation | target | usable for |
|---|---|---:|---:|---|---|---|---|
| STW / Faces95, официальные производные face crops | Да, переиспользован STW ZIP | 72 | 1 440 | Ровно 20 на человека; 13 680 пар | Реальные разные кадры; номера camera/light/session отсутствуют | source identity; MST остаётся отдельной разметкой STW | RGB capture-invariance candidate; upstream rights пока не подтверждены |
| Georgia Tech, оригинальный авторский архив | Да, ZIP + 750 face boxes | 50 | 750 | Ровно 15; 5 250 пар | 749 EXIF Sony HANDYCAM; 30 людей имеют ≥2 EXIF-даты; освещение описано автором, но не размечено по кадрам | identity и capture timestamp; не Lab | Более содержательные сведения о capture; явное разрешение на обучение не восстановлено |
| AT&T / ORL, официальный Cambridge архив | Да | 40 | 400 | Ровно 10; 1 800 пар | Автор описывает небольшие изменения света и времени; без точных меток | identity, grayscale capture | Только ограниченный ахроматический auxiliary/control с атрибуцией |
| Multi-PIE | Нет | 0 получено | 0 | Не получены | Официальный путь заказа не доставляет данные | — | BLOCKED_EXTERNAL_DISTRIBUTION |
| Essex Faces95, оригинальные кадры | Нет | 0 получено | 0 | Доступна только производная STW выше | Официальный сервер 502/timeout | — | BLOCKED_OFFICIAL_HOST для оригинала |
| Extended Yale B | Нет | 0 получено | 0 | Не получены | Официальный cropped ZIP 404; страница timeout | — | BLOCKED_OFFICIAL_HOST |

Дата проверки: 2026-09-15. Это новые фактически полученные данные и их квалификация, не обучение. C_REBASE_V1, production и inference не изменены. Полный пользовательский список 60 источников в доступном репозитории/рабочей папке не найден; принадлежность этих fallback-наборов этому списку не утверждается. Сетевые mirrors и обходы доступа не использовались.

STW / Faces95 уже находится внутри полученного официального STW release, поэтому повторной загрузки не было. Это не 1 440 дополнительных к общему STW снимков. Все 1 440 сопоставлены с CSV строками `dataset == Faces 95`, уникальны по SHA256 и декодированы: RGB 300×300, действительно различные каналы; EXIF и ICC отсутствуют. Стабильный identity берётся из `tokens`, порядковый capture — из исходного имени кадра. Имена и исходные пути сохранены только в private manifest. Числа 1–20 не превращаются в illumination/session labels. Это авторские MediaPipe crops, а не исходные Essex frames; связь crop с координатами оригинала не восстановлена. [Официальный STW репозиторий](https://github.com/vitorpmh/STW).

Технически Faces95 — первый кандидат на RGB repeated-capture objective: больше независимых identities, одинаковые 20 кадров на человека. Но без идентификаторов света нельзя изолировать изменение освещения от позы, выражения и crop; identity objective также не доказывает восстановление intrinsic skin reflectance. Оригинальная Essex страница недоступна. STW README требует соблюдать лицензии исходных наборов и не выдаёт им новую общую лицензию; исследовательские права Essex остаются непроверенными. Это неопределённость условий, а не установленное требование нового платного доступа или обязательного договора владельца.

[Автор Georgia Tech](https://www.anefian.com/research/face_reco.htm) предоставляет 50-персонный набор с изменениями освещения, выражения, наклона и масштаба; [README](https://www.anefian.com/research/GTDB_README.txt) однозначно задаёт image↔person и image↔face-box. Все 750 JPEG 640×480 успешно декодированы; 750 boxes лежат внутри кадров. Метаданные 749 файлов указывают `SONY HANDYCAM`, EXIF ColorSpace=1 (заявленный sRGB), но конкретная модель/единица камеры и параметры exposure/WB отсутствуют; один файл без EXIF. Разметка приборного цвета отсутствует.

В Georgia Tech найдено 16 EXIF-дней: у 20 людей известен один день, у 27 — два, у 3 — три. Дата — proxy, а не официальный session ID; авторское описание 2–3 sessions нельзя подменять этим подсчётом. Есть 3 767 пар с одной известной датой, 1 469 с разными, 14 с недостающей датой. Ни сам факт общей identity, ни близость даты не доказывают постоянство физического цвета кожи. Авторская страница и README дают download и описание исследований, но явного use grant не содержат: dataset-specific rights не объявлены подтверждёнными. Изображения не публикуются и обучение на них не запускалось.

[Официальный AT&T/Cambridge архив](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) поставляется с просьбой об атрибуции AT&T Laboratories Cambridge; вложенный README также называет Olivetti Research Laboratory. Получены все 400 PGM 92×112, 8 bit grayscale, 400 уникальных SHA256. После перевода в RGB все три канала равны: новых цветовых измерений это не создаёт. Его небольшой световой разброс пригоден лишь для ахроматического контроля; он не равноценен Multi-PIE или цветному lighting dataset. Стандартизованная коммерческая лицензия не заявляется.

Наблюдаемый разброс измерен до какого-либо обучения. Для Georgia Tech использована центральная половина авторского face box; для STW — центральные 150×150 crop; для ORL — центральная половина кадра. Эти прямоугольники содержат некожные пиксели и служат только проверке различий кадров. Медиана encoded RGB масштабирована в [0,1]; brightness proxy = 0.2126R + 0.7152G + 0.0722B, без заявления линейной физической яркости.

| Набор / пары | N pairs | RGB-distance mean / median / p95 | Absolute brightness-proxy difference mean / median / p95 |
|---|---:|---|---|
| STW / Faces95 | 13 680 | 0.05477 / 0.03529 / 0.15394 | 0.02724 / 0.01738 / 0.07758 |
| Georgia Tech, все | 5 250 | 0.07542 / 0.04349 / 0.27729 | 0.04078 / 0.02326 / 0.15328 |
| Georgia Tech, один EXIF-день | 3 767 | Не отдельная метрика | 0.02357 / 0.01792 / 0.06301 |
| Georgia Tech, разные EXIF-дни | 1 469 | Не отдельная метрика | 0.08511 / 0.06506 / 0.21825 |
| ORL | 1 800 | 0.07093 / 0.04755 / 0.21736, replicated gray RGB | 0.04095 / 0.02745 / 0.12549 |

Больший междневный разброс Georgia Tech — описательный результат. Он смешивает освещение, положение, выражение, обработку камеры и возможные изменения самой кожи. Это не приборный noise floor, не причинная оценка света и не показатель качества модели.

Все bytes/URLs/SHA256/CRC сохранены в `acquisition.json`, измеренные counts и distribution — в `qualification.json` и `stw_faces95_qualification.json`, точные blockers — в `blockers.json`. Официальный Multi-PIE distribution link нативно вернул 403, а browser fetch перенаправился на общий Wellspring search; оплаты, заявок и договоров не было.

Воспроизведение из корня репозитория (Python 3.12.14; NumPy 2.3.5; Pillow 12.3.0; стандартный urllib):

```bash
python scripts/data_access/repeated_capture_acquire.py --receipt data/public/repeated_capture_access/acquisition_restoration.json
python scripts/data_access/repeated_capture_qualify.py
python scripts/data_access/repeated_capture_stw_faces95.py
```

Первая команда требует новый receipt path: существующий receipt никогда не перезаписывается. Три архива сверяются с жёстко зафиксированными размерами и SHA256 первой загрузки, затем с ZIP CRC. Несоответствие cache или новой загрузки приводит к отказу без замены существующего файла. Первичный `acquisition.json` с фактическими HTTP status/final URLs сохранён неизменным; cache recheck не выдумывает новый HTTP response. Для следующих повторов указывайте другой новый `--receipt` либо выполняйте только offline-проверку:

```bash
python scripts/data_access/repeated_capture_acquire.py --verify-cache-only
python scripts/data_access/check_repeated_capture_acquire.py
```

Offline QA (`cache_verification.json`, `acquisition_qa.json`) проверил все 6 имеющихся source files, совпадение 3 archive pins с первым receipt, отказ для испорченного cache того же размера, отсутствующего cache и испорченной новой загрузки, а также запрет перезаписи primary receipt. Отрицательные проверки выполнялись на временных fixtures, без изменения реальных данных и без сети.

Третья команда основного блока не обращается к сети и требует ранее скачанные `data/public/stw_access/images.zip` и `all_annotated_data.csv`. Оригинальные/производные лица, EXIF timestamps, исходные токены и row-level manifests остаются под gitignored `data/public/`; публичные результаты содержат только агрегаты и обезличенные группы. Никакой target Lab, pseudo-Lab, новой архитектуры или инструмента оценки цвета здесь не создаётся.
