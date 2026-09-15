# ENCoDE — фактическая проверка доступа

**BLOCKED_CREDENTIALS_DUA.** Получены шесть публичных документов/страниц; защищённые participant metadata, изображения и приборные строки не получены. Официальная карточка отвечает HTTP200, официальный files endpoint HTTP403. Предыдущий проект уже обнаружил этот gate; текущая проверка подтверждает его, а не выдаёт ENCoDE за новый скачанный dataset.

Для доступа владелец должен оформить credentialing PhysioNet, требуемое CITI Data or Specimens Only Research и лично согласиться с DUA проекта. Агент не использовал логин, не принимал условий и не отправлял писем. Лицензия запрещает делиться доступом; пароль передавать не нужно.

[Публичная карточка](https://physionet.org/content/encode-skin-color/1.0.0/) описывает processed smartphone crops, раздельные iPhone/Android папки и имя `person_id_location_id`. Связь с measurement проходит через PERSON/VISIT_OCCURRENCE/MEASUREMENT/CONCEPT. Для image features путь указан в MEASUREMENT_SOURCE_VALUE. Ключ конкретного capture и однозначность связи с каждым instrument reading на фактических строках **не проверены**, поскольку они закрыты.

В release описаны десять image sites, включая лоб; часть ладонных/подошвенных изображений исключена из соображений биометрической приватности. Нельзя считать наличие их RGB feature row наличием фотографии. Участок лба и обычное полное селфи — разные входы.

Детальный контракт, неизвестные параметры и условия отбраковки неоднозначных пар: [contract.documented_not_row_verified.json](contract.documented_not_row_verified.json). Observer/illuminant должны быть подтверждены отдельно для каждого прибора. MST, derived image RGB/LCH и настоящие instrument coordinates не смешиваются.

Источник кода: [официальный tutorial](https://github.com/aiwonglab/ENCoDE_tutorial), MIT; данные имеют отдельные ограничительные условия PhysioNet. Никакой численной точности или обучения на ENCoDE в этом этапе нет.

Выполненная команда регистрации уже скачанных файлов:

```bash
python scripts/data_access/encode_public_metadata.py --output data/public/encode_access --receipt docs/data/access_2026_09_15/encode/download_receipt.json --use-existing
```

Для новой проверки без перезаписи текущего evidence используйте другие output/receipt paths. Это проверка официальных публичных документов, а не загрузчик защищённых данных. Сохранены source URLs, HTTP statuses, sizes и SHA256.
