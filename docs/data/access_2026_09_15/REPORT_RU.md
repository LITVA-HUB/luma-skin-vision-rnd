| dataset | downloaded | people | images | repeated captures | camera/illumination variation | target | usable for |
|---|---|---:|---:|---|---|---|---|
| ENCoDE | Только 6 публичных документов; participant data нет | 0 получено | 0 | Не получены | Документированы iPhone SE 2020 / Pixel 4a; строки закрыты | Instrument readings потенциально; сейчас 0 проверенных пар | BLOCKED_CREDENTIALS_DUA |
| Multi-PIE | Нет | 0 | 0 | Не получены | Официальный путь заказа не доставляет архив | Identity / capture, если получен | BLOCKED_EXTERNAL_DISTRIBUTION |
| Georgia Tech | Да, оригинальный ZIP + face boxes | 50 | 750 RGB | 15/человека; 5 250 пар | 749 EXIF Sony HANDYCAM; 30 людей сняты в ≥2 EXIF-даты; свет описан автором, без frame labels | Identity; timestamp, не Lab | RGB repeated-capture сигнал получен; явные условия использования не восстановлены |
| AT&T / ORL | Да, официальный архив | 40 | 400 grayscale | 10/человека; 1 800 пар | Автор описывает изменения света/времени; точных frame labels нет | Identity, не Lab | Ограниченный ахроматический auxiliary/control с атрибуцией |
| STW, доступная часть | Да, 636 672 248 B ZIP + metadata/splits | 1 878 source-person keys | 14 278 full-face; ещё 14 205 производных skin-only вариантов | 847 source-person групп имеют >1 кадра | Разные реальные исходные фото; camera/light/session labels отсутствуют | Human per-person MST 1–10 | Apparent-tone auxiliary; source-specific права и cross-source identity ограничения |
| Faces95 внутри STW | Да; повторно не скачивался | 72 | 1 440 RGB, входят в STW выше | 20/человека; 13 680 пар | RGB crops 300×300; нет EXIF/точных illumination/session IDs | Identity; MST отдельно, не Lab | RGB repeated-capture кандидат; upstream права не подтверждены |
| UMINHO-HSFD | Сейчас 0 кубов; восстановлен точный список прежних 19 TRAIN-источников | 0 получено | 0 | Не получены | Официальный Figshare API/page: HTTP 403 | Spectral reflectance, если оригиналы восстановлены | BLOCKED_SOURCE_HTTP; P2 не требуется для новой подготовки |

Это результат получения и проверки данных на 15 сентября 2026, а не новое обучение или улучшение ΔE00. DAST и CHROMA-FIT остаются **BLOCKED_EXTERNAL по сообщению владельца**; в этом этапе их доступ не проверялся и запросы не отправлялись. C_REBASE_V1 и текущий inference сохранены без изменений.

Счётчик STW — число уникальных `(source dataset, person token)`, не доказанное число биологически разных людей во всех источниках. Faces95 уже включён в STW и не суммируется с ним. Skin-only варианты не являются дополнительными независимыми съёмками. Пересечения identities между разными источниками неизвестны.

## ENCoDE: точный текущий блокер и контракт

[Официальная страница ENCoDE](https://physionet.org/content/encode-skin-color/1.0.0/) доступна, files endpoint возвращает HTTP 403. Получены публичные описание, DUA, лицензия и tutorial, но **0 фактических image/instrument pairs**. Это подтверждение прежнего access gate, не новый скачанный приборный dataset.

**Действие владельца:** оформить доступ в разделе Files ENCoDE через собственный credentialed PhysioNet account, требуемое CITI Data or Specimens Only Research и личное согласие с project DUA. Передавать пароль не нужно; агент не принимал договоров и не использовал чужие credentials. Разрешённую среду использования полученных данных нужно соблюдать согласно индивидуальному доступу.

Документированная связь, пока не проверенная на реальных строках:

| Поле | Что установлено / что отсутствует |
|---|---|
| image | Обработанный skin-site crop; папки iPhone/Android; имя `person_id_location_id`. Это не обещание исходного full-face JPEG. |
| person | PERSON.person_id. Нужен точный join с metadata, а не распознавание личности по пикселям. |
| body site | Десять выпущенных image sites, включая forehead. Список в contract JSON. Ventral biometric-sensitive снимки частично исключены. |
| phone/device | iPhone SE 2020 и Google Pixel 4a документированы; фактическая построчная проверка ещё невозможна. |
| capture/session | VISIT_OCCURRENCE связывает admission; admission не объявляется уникальным capture. Точные acquisition/time keys не получены. |
| instrument | Delfin SkinColorCatch, Konica-Minolta CM-700d, Variable Spectro 1 Pro; каждый прибор остаётся отдельным источником. |
| instrument measurement | MEASUREMENT + CONCEPT; concept template `SKINTONE@<Location>_<device manufacturer>.<measure>`. Для image features путь в MEASUREMENT_SOURCE_VALUE. |
| units/color convention | Нельзя заимствовать D65/10° MSKCC. Реальные unit concepts, настройки observer/illuminant, acquisition-specific triplets предстоит проверить на выпущенных строках. |

Join принимается только при однозначных person + visit + exact site + instrument + acquisition/time и согласованном L/a/b одного измерения. Один person ID без site/acquisition недостаточен. Наличие image-derived RGB/LCH features не считается наличием фотографии или приборного Lab. MST и instrument coordinates остаются разными target types.

Подробности: [ENCoDE report](encode/README_RU.md), [documented-only contract](encode/contract.documented_not_row_verified.json), [actual HTTP probe](encode/access_probe.json).

## Полученные реальные repeated captures

После недоставляющего данные официального Multi-PIE ordering path выполнены ограниченные проверки подходящих источников, затем получены реальные файлы Georgia Tech и ORL. Исторический пользовательский «список 60» в доступных workspace/repo не найден; принадлежность выбранных fallback-наборов этому списку не утверждается. Нового общего поиска или обходов закрытых наборов не было.

Для текущих файлов **Georgia Tech даёт наиболее содержательные capture metadata**: 750 цветных фотографий, 750 валидных face boxes, 50 стабильных source identities; 749 EXIF записей. Для 30/50 людей есть ≥2 разных capture dates. Это удобнее для отделения cross-day пар, чем одни последовательные номера Faces95. Но EXIF-дата — proxy, не официальный session/light ID. В архиве и на авторской странице не восстановлен явный текст разрешения использования, поэтому это технически полученный сигнал с неуточнёнными условиями, а не подтверждённый unrestricted training corpus.

В Georgia Tech 3 767 same-day, 1 469 cross-day и 14 date-unknown пар. Средняя абсолютная разница encoded RGB brightness равна **0.023573** для same-day и **0.085108** для cross-day пар. Это описательная вариативность face box, который включает не только кожу; она смешивает позу, выражение, свет, обработку камеры и возможные изменения кожи. Она не доказывает причинное влияние освещения и не является Lab noise floor.

Faces95 в официальном STW выпуске даёт 1 440 различных RGB crops /72 группы по20. Exact image/identity join и декодирование проверены. Capture ordinal не превращается в illumination label. Оригинальный Essex host сейчас недоступен; исходные права не квалифицированы заново. AT&T/ORL даёт ещё 400 проверенных grayscale кадров /40 людей, пригодных только как ограниченный ахроматический auxiliary/control, с предусмотренной автором атрибуцией.

Все 750 GT +400 ORL +1 440 Faces95 изображений декодированы без ошибок. Фото и исходные IDs остаются локально в ignored data directories. [Измерения, лицензирование и receipts](repeated_capture/REPORT_RU.md). Author sources: [Georgia Tech](https://www.anefian.com/research/face_reco.htm), [AT&T/ORL](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html).

Ни один из этих наборов не даёт absolute instrument Lab. Positive same-person pairs могут проверять устойчивость к capture, но сами по себе не доказывают intrinsic reflectance: encoder может запоминать форму лица или выражение. Точных пар одного участка кожи под известным спектром света здесь нет.

## STW: новый опубликованный сигнал и его пределы

Получен [официальный выпуск автора STW](https://github.com/vitorpmh/STW), commit `172e4906de9e91d7cee1090e63a34db95fe03f69`. Его текущий README содержит доступ к файлам, хотя прежнее описание статьи говорило о будущем release. **28 483/28 483 JPEG декодированы и однозначно связаны с CSV/MST**, failures =0. Основные captures:14 278; дополнительные варианты:14 205; отсутствуют 73 skin-only варианта. Скачаны annotation CSV и14 author split CSV.

В доступной paired части author train содержит11 580 кадров, test2 698. Source-person overlap train/test=0; глобальная биологическая непересекаемость между исходными наборами не доказана. Splits проверены структурно; никакой model selection или оценки по ним не было.

MST — human per-person category, распространённая на снимки человека. Это auxiliary apparent-tone signal, без измеренного спектра/CIELAB. 42 312 строк основной annotation CSV не означают42 312 скачанных фотографий: CASIA/CelebA есть только в metadata. В этой поставке их pixels отсутствуют.

Права различаются по источникам. Для1 031 FairFace изображения заново подтверждена исходная image license CC BY4.0. FEI разрешает research use и запрещает redistribution; author-linked STW release не отменяет эти ограничения. У Essex/LFW точный доступный текст прав не восстановлен. Общая STW annotation/release license ещё WIP; README делегирует условия исходным datasets. Это не подтверждение единого unrestricted или коммерческого разрешения на весь ZIP. [Source-by-source таблица, точные counts и joins](stw/REPORT_RU.md).

## UMINHO: восстановление прежнего протокола, без нового эксперимента

Восстановлены exact19 TRAIN filenames, sizes и SHA256 из исторического source lock: ожидается1 975 360 986байт. Текущих `.mat` в кэше нет; Figshare возвращает403; сейчас декодировано0кубов. Этот статус относится к текущему сетевому доступу, не к отсутствию набора в мире.

Подготовлен проверяющий hashes загрузчик и вновь рассчитаны интеграционные матрицы D65/CIE1931 2° и D65/CIE1964 10° с явной экстраполяцией вне400–720нм. Математическая проверка идеального отражателя пройдена, **реальные cube rows отсутствуют**. Производный D65/CIE2006 10° JPEG не считается независимой RGB-фотографией и не смешивается с CIE1964 10°.

Потерянные P2 weights не мешают заново подготовить supervision из оригиналов. Точное повторение старого P1/P2 дополнительно требует прежних Seg1/ROI/masks; это отдельное требование репродукции, не блокер проекта. [Полный UMINHO отчёт](uminho/REPORT_RU.md).

## Выполненные команды и сохранность

Команды выполнялись из корня repo. Аналитический Python:3.12.14, NumPy2.3.5, Pillow12.3.0; acquisition использует urllib. Существующие завершённые caches повторно проверялись и переиспользовались.

```bash
python scripts/data_access/encode_public_metadata.py --output data/public/encode_access --receipt docs/data/access_2026_09_15/encode/download_receipt.json --use-existing
python scripts/data_access/stw_acquire.py --root data/public/stw_access
.venv-c-rebase/bin/python scripts/data_access/stw_audit.py --private-root data/public/stw_access --report-dir docs/data/access_2026_09_15/stw
python scripts/data_access/repeated_capture_acquire.py --verify-cache-only
python scripts/data_access/repeated_capture_qualify.py
python scripts/data_access/repeated_capture_stw_faces95.py
python scripts/data_access/uminho_restore.py inventory
python scripts/data_access/uminho_restore.py contract
python scripts/data_access/uminho_restore.py audit
```

Подробные URLs, sizes, HTTP status, SHA256 и команды находятся в source-specific reports/receipts. `MANIFEST_SHA256.json` закрепляет публичные результаты и acquisition/audit scripts. Raw image/person mapping хранится только локально в ignored `data/public/`; GitHub содержит код, агрегаты, контракты и hashes, без фотографий и прямых identifiers.

Первое получение repeated-capture архивов выполнено до дополнительного QA загрузчика. После QA закреплены ожидаемые sizes/SHA256 трёх ZIP и запрет перезаписи первичной сетевой квитанции; `--verify-cache-only` проверяет текущие файлы без сети. Для отсутствующего кэша в новом clone используется `python scripts/data_access/repeated_capture_acquire.py --receipt data/public/repeated_capture_access/acquisition_new.json`; существующий receipt сохраняется, для следующей отдельной загрузки нужен новый путь. Неверный hash, даже при корректном ZIP CRC, прекращает приём файла.

Попытка долговременного приватного сохранения четырёх полученных ZIP завершилась **до загрузки** ошибкой авторизации runtime storage. Поэтому сохранение этих фотоархивов вне текущего workspace **не подтверждено**. Локальные завершённые ZIP доступны; публичные receipts и воспроизводимые download-команды публикуются в GitHub. Фотографии в Git не добавлялись ради обхода этой ошибки. Срок жизни локального кэша не гарантируется.

## Решение текущего этапа

**Новые полезные сигналы фактически получены:** RGB repeated-person captures и human MST-linked face images. Для camera/light supervised objective нет точных per-frame illumination labels; для нового instrument head получено **0 новых реальных image/reference pairs**. Права не сведены к одной общей лицензии. Это достаточное основание продолжать квалификацию этих реальных файлов, но не заявлять доказанную intrinsic representation или улучшение цвета.

Новые архитектуры и обучение в этом этапе не запускались. C_REBASE_V1 остаётся frozen MSKCC baseline; production C и EasyPortrait inference неизменны. Никаких pseudo-Lab или MST→Lab targets не создано. Точность на селфи не повышалась и не измерялась. Уточнение ENCoDE требует пользовательского credentialed access; UMINHO не блокирует работу с уже полученными repeated captures/STW. DAST и CHROMA-FIT остаются вне активной очереди до сообщения владельца.
