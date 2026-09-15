# STW: фактически полученный частичный выпуск, 2026-09-15

| dataset | downloaded | people | images | repeated captures | camera/illumination variation | target | usable for |
|---|---|---:|---:|---|---|---|---|
| STW: доступная часть официального выпуска | Да, 636 672 248 B ZIP + CSV и 14 split CSV | 1 878 source-person keys | 14 278 полно-лицевых crops; дополнительно 14 205 skin-only вариантов тех же кадров | 847 групп имеют >1 исходного кадра | Реальные вариации исходных наборов; camera/exposure/illumination/session в CSV отсутствуют | Human MST 1–10, annotation per person | Apparent-tone auxiliary; права проверяются отдельно по исходному набору |

`people` здесь означает уникальный `(dataset, tokens)` внутри выпуска, а не доказанное число биологически разных людей во всех источниках. Между LFW/FairFace и другими источниками нельзя исключить совпадение людей. Разбиение по глобальной identity ещё не квалифицировано.

## Что изменилось по сравнению с прежним inventory

В [официальном arXiv abstract](https://arxiv.org/abs/2603.02475) всё ещё указано, что код и данные появятся позже. Проверка фактического GitHub автора обнаружила [vitorpmh/STW](https://github.com/vitorpmh/STW), commit `172e4906de9e91d7cee1090e63a34db95fe03f69`. Его README ведёт на публичный Google Drive с `images.zip`, `all_annotated_data.csv` и splits. Это новый полученный частичный выпуск относительно прежнего аудита, не повторный список возможных наборов.

README прямо предупреждает, что текущий выпуск ещё не соответствует результатам статьи. В репозитории 30 файлов, но исходных фотографий и CSV там нет: они получены из связанного авторами Drive. В ZIP **нет** фотографий CASIA Face Africa, CASIA V5 или CelebA; их gates не обходились.

## Проверенные файлы и соответствие image → label

- ZIP SHA256: `6bca3b5074c115d7f871a58414f0b24654e9ff2c90461ed901d4090dcd5d0e95`.
- `all_annotated_data.csv`: 3 818 026 B, SHA256 `ef98d1a0a23ca338b150542a2b596146f76308e58708276c0cf07314fd9a0d57`.
- CSV содержит **42 312** уникальных путей и **3 587** source-person keys. Это отличается от 42 313 / 3 564 в статье; числа статьи не присваиваются скачанным файлам.
- Проверены CRC архива и фактическое декодирование **28 483 JPEG**. Все 28 483 однозначно сопоставлены CSV и совпадают с MST-категорией папки; повреждённых, неоднозначных или несопоставленных файлов **0**.
- 14 278 full-face файлов — RGB JPEG **300×300**, обработанные авторским MediaPipe crop; это не camera-original файлы.
- 14 205 skin-only файлов — дополнительные производные представления, **не** ещё 14 205 независимых captures. Для 73 full-face кадров skin-only варианта нет.
- Побайтовых совпадений между разными исходными строками не найдено. Это не исключает другого crop/encoding одного кадра или одного человека.

Точное соединение восстановлено из авторских processing scripts:

```text
full_face filename = new_tokens + stem(paths) + "_face_1.jpg"
skin_only filename = new_tokens + "_" + stem(paths) + ".jpg"
MST = integer CSV['class']; папка класса должна совпадать
person_group = (dataset, tokens)
```

Локальный manifest `data/public/stw_access/image_pairing.private.json` содержит source path, image member, class, group и SHA256 каждого JPEG. Он не публикуется, поскольку исходные имена могут содержать прямые идентификаторы. Публичны только агрегаты и SHA256 файлов.

## Реальная доступная часть и права

| Исходный набор в CSV | Full-face кадров получено | Source-person keys | Группы с повторениями | Правовой/технический статус |
|---|---:|---:|---:|---|
| FairFace | 1 031 | 1 031 | 0 | [Официальный dataset README](https://github.com/joojs/fairface) указывает CC BY 4.0. Наиболее явно разрешённая часть для apparent-MST auxiliary. Attribution обязателен. |
| Faces 95 | 1 440 | 72 | 72, ровно 20 кадров | Pixel/label pairing проверен. Точная текущая лицензия Essex квалифицируется в отдельном repeated-capture отчёте; не выдаётся за автоматически разрешённую коммерческую часть. |
| Faces 94 | 3 040 | 152 | 152, ровно 20 кадров | Аналогично: оригинальное предоставление Essex и права требуют отдельной фиксации. |
| Brazilian Faces / FEI | 2 782 | 200 | 200, 13–14 кадров | [Официальный FEI source](https://fei.edu.br/~cet/facedatabase.html) разрешает research use, запрещает распространение базы. Авторский STW ZIP не снимает исходное ограничение; для чистой цепочки происхождения перед применением этой части предпочтительно получить оригинал у FEI. |
| LFW | 5 985 | 423 | 423, 5–530 кадров | Фото/labels получены; актуальная страница правообладателя недоступна из среды. Не включено в строгий список частей с заново подтверждёнными условиями. |

**Строго подтверждённая исходная image-license часть: 1 031 FairFace кадр.** Это не утверждение, что остальные 13 247 кадров запрещены: их лицензирование квалифицируется отдельно. STW README пока не даёт отдельного готового текста лицензии для общего выпуска и предписывает соблюдать лицензии исходных наборов. Отдельное коммерческое право на annotations/combined release не подтверждено. Никакие фотографии или исходные идентификаторы не публикуются.

CASIA Face Africa (24 655 metadata rows), CASIA V5 (2 500) и CelebA (823) в annotations присутствуют, но соответствующих пикселей в полученном ZIP нет. Они не учитываются в downloaded images или usable training pairs.

## Семантика нового supervision signal

По [авторской статье, раздел 3](https://arxiv.org/html/2603.02475v1#S3), основную MST-разметку всем людям дал один human annotator; два дополнительных annotators проверяли подвыборку 1 000 людей. Одна person-category распространяется на изображения этого человека. В CSV нет отдельных оценок annotators или индивидуального instrument reading. Это **видимая категория кожи на фотографиях с субъективной интерпретацией**, а не измерение reflectance/intrinsic Lab. MST не преобразуется в Lab.

CSV имеет только `paths, tokens, dataset, new_tokens, number_of_photos, class`. Конкретные camera, exposure, illuminant, session и anatomical ROI поля не выпущены. Нельзя приписывать их по имени человека, цвету пикселей или MST class. Raw поляризация/спектр/приборные значения отсутствуют.

## Разделения и ограничения оценки

14 авторских individual split CSV скачаны и сохранены без перезаписи. В полученной paired части:

- train: 11 580 кадров / 1 508 source-person keys;
- test: 2 698 кадров / 370 source-person keys;
- holdout_train: 9 406 / 1 190;
- holdout_val: 2 174 / 318.

Полные авторские train+test CSV содержат 29 378 уникальных source paths / 3 564 source-person keys, а не все 42 312 строки основной annotation CSV. В split manifests на 12 896 строк CASIA Face Africa и 38 строк Faces94 меньше; missing pixels не скачивались. Все 14 278 доступных full-face кадров покрыты train/test ровно один раз. Для всех 14 split CSV проверено: ноль отсутствующих путей, MST-конфликтов и source-person-конфликтов относительно основной CSV.

Пересечение source-person keys между train/test, holdout_train/val и внутри каждого из пяти авторских folds равно нулю. Глобальное cross-source identity/near-duplicate пересечение **не доказано отсутствующим**; эти splits не объявляются готовым независимым facial instrument test. В настоящем этапе splits проверялись только структурно, модели не обучались и качество MST не измерялось.

## Воспроизводимые команды

```bash
# Оригинальные author-linked assets; существующие совпадающие хэши используются повторно.
python scripts/data_access/stw_acquire.py --root data/public/stw_access

# Реальный JPEG decode, exact image/label join, подсчёт групп и split overlap.
.venv-c-rebase/bin/python scripts/data_access/stw_audit.py \
  --private-root data/public/stw_access \
  --report-dir docs/data/access_2026_09_15/stw
```

У `stw_acquire.py` зафиксированы размер и SHA256 каждого из 16 файлов. При отличии файл не перезаписывается молча. Единственный обработанный Drive prompt — обычное предупреждение о невозможности antivirus scan большого ZIP; login, consent, EULA или dataset-access approval не запрашивались и не подтверждались.

`inventory.json` содержит все численные проверки, `file_receipts.json` — хэши локальных файлов. `C_REBASE_V1` и production inference не изменялись. Нового обучения в этой проверке нет.
