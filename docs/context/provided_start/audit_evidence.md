# Аудит: проверяемые фрагменты

[PROJECT FACT] Источник: предоставленный ZIP Luma-2.0-main (1)(1).zip. Нумерация строк относится к распакованному исходному файлу.

Область: сначала документация и зависимости/лицензия, затем точечная проверка vision/provider/API. Полный исходный код и UI не аудировались. Полная тестовая матрица не запускалась.

Выполнен только smoke-check встроенного фото на CPU; отдельный JSON содержит фактический результат. На RTX 4060 ничего не измерялось.


## `README.md:21–24`

```text
  21 **Нативный iPhone-клиент · SwiftUI · FastAPI · 240 демонстрационных SKU**
  22 
  23 Локальный пилот для «Золотого Яблока». Каталог и сценарии на экранах —
  24 демонстрационные; реальные магазинные интеграции ещё не подключены.
```


## `README.md:177–190`

```text
 177 ## Дальше — из пилота в продукт
 178 
 179 Для внешнего запуска остаются реальные catalog/auth/checkout/SMS/push и analytics
 180 интеграции, production-инфраструктура, проверка восстановления, оценка фото
 181 и рекомендаций на репрезентативных данных, legal-страницы и матрица устройств.
 182 Строгий live OpenRouter gate пока не закрыт. Прошлые прогоны не подтверждают
 183 текущую сборку; контракт провайдера не означает подключённую интеграцию.
 184 
 185 **[Статус](docs/CURRENT_STATUS.md)** · **[QA](docs/QA_REPORT.md)** ·
 186 **[Критерии релиза](docs/PRODUCTION_RELEASE_CRITERIA.md)** · **[Live-пилот](docs/LIVE_PILOT.md)**
 187 
 188 ---
 189 
 190 <p align="center"><strong>ЗОЛОТОЕ ЯБЛОКО / BEAUTY ID</strong><br /><sub>Internal · Proprietary · Права на бренд и ассеты требуют отдельной проверки перед внешним распространением.</sub></p>
```


## `docs/ARCHITECTURE.md:96–119`

```text
  96 ## Reference recommendation core
  97 
  98 All automated product decisions use one deterministic provider-independent
  99 pipeline:
 100 
 101 ```text
 102 CatalogProvider.get_snapshot() (snapshot-owned deeply immutable Products)
 103   -> CatalogIndex.build() (complete fail-closed validation + normalized facts)
 104   -> RecommendationContext (Beauty ID + bounded focus + supported filters
 105                              + authoritative mode + seen SKU history)
 106   -> EligibilityPolicy (hard rejections)
 107   -> RecommendationRanker (typed lexicographic FitVector)
 108   -> RecommendationEngine (limited, grounded decision)
 109   -> RecommendationPresenter (existing public DTO)
 110 ```
 111 
 112 The pure engine receives only a context and catalog index. It does not read the
 113 database, provider, environment, wall clock, HTTP, FastAPI, or an LLM. Identical
 114 context, catalog version, and `reference-v3` engine semantics therefore produce
 115 the same canonical-SKU order for structurally valid unique-SKU snapshots,
 116 regardless of provider response order or SKU prefix. Malformed duplicate-SKU
 117 snapshots fail closed before ranking. Routine builder, deterministic
 118 advisor compiled tasks, scan fallback,
 119 win-back, and PDP rationale reuse the same eligibility/ranking boundary instead
```


## `docs/AI_AGENT_AND_BACKEND_ARCHITECTURE.md:98–112`

```text
  98 1. Preflight в `advisor_safety.py` проверяет медицинский риск. Запрос на
  99    диагноз/лечение получает безопасный ответ без вызова LLM.
 100 2. Роут читает Beauty ID, режим, историю текущей сессии, корзину и набор.
 101    `advisor_references.py` формирует серверные ссылки на допустимый контекст.
 102    Произвольное клиентское `current_skus` не выбирает полномочия.
 103 3. `advisor_privacy.py` создаёт ограниченный `AdvisorProviderInput`: очищенное
 104    сообщение, очищенные предыдущие реплики, сводку ссылок и версию онтологии.
 105    Профиль, реальные account/session IDs, цены и полные карточки не отправляются
 106    как контекст. Явно написанный пользователем бюджет сохраняется для разбора
 107    ограничения. Приватные значения и служебные маркеры очищаются.
 108    Фото в этом вызове не участвует.
 109 4. Провайдер возвращает `AdvisorIntent` из `advisor_intent.py` /
 110    `advisor_contracts.py`. Схема и семантическая проверка различают разговор,
 111    уточнение, подбор и предложение изменения. Корректный JSON ещё не даёт
 112    разрешения выполнить содержащуюся в нём задачу.
```


## `docs/AI_AGENT_AND_BACKEND_ARCHITECTURE.md:188–195`

```text
 188 ## Остальные сценарии
 189 
 190 - Фото: `routes/scan.py` → `scan.py` → `services/face_vision.py`.
 191   Проверяются согласие, размер и формат. YuNet ищет лицо, измерение цвета
 192   предлагает видимую глубину тона. Освещение и ракурс влияют на результат.
 193   Тип кожи, диагноз и гормоны по фото не определяются. Сохраняются
 194   подтверждённое предпочтение и ограниченная история, без фото, landmarks
 195   и исходных цветовых образцов.
```


## `docs/CURRENT_STATUS.md:96–100`

```text
  96 ## Расширение каталога и обновление фотосканера — 2026-09-06
  97 
  98 Каталог расширен до **240 синтетических SKU** (239 в наличии): добавлены
  99 146 вариантов из 69 демонстрационных семейств в 16 категориях. Новые свойства
 100 явно заданы как тестовые данные; старые 94 записи и их UNKNOWN-статус отдушки
```


## `docs/PRODUCTION_GAPS.md:31–46`

```text
  31 - Deterministic `reference-v1` recommendation core with fail-closed catalog
  32   validation and hard eligibility, provider-order/SKU-prefix independence,
  33   shared recommendation/routine/advisor/scan/win-back/PDP policy, exact routine
  34   budgets, executable golden cases, generated invariants, and intentional
  35   abstention when verified metadata is insufficient.
  36 - Atomic production catalog snapshot contract with conditional upstream token
  37   validation, immutable last-good publication, and request-path snapshot/index
  38   reuse. This is a provider boundary only; no Golden Apple adapter is connected.
  39 
  40 ## Remaining before real TestFlight/App Store ❌
  41 
  42 1. Connect production auth provider (currently dev-login + guest + local tokens).
  43 2. Implement `GoldenAppleCatalogProvider`, raw retail record to `Product`
  44    mapping, conditional upstream revision/ETag validation, and licensed product
  45    image CDN integration (the current 94-SKU set is synthetic development
  46    data). The vendor token must populate `upstream_revision` separately from the
```


## `docs/recommendation-ranking-policy.md:84–89`

```text
  84 `match_score = floor(100 × matched_criteria / requested_criteria)`, or zero when
  85 there are no applicable criteria. `known_criteria` separately describes available
  86 catalog evidence. Missing metadata is not a confirmed match. The score measures
  87 coverage of the applicable structured criteria, not purchase probability, model
  88 confidence, guaranteed effectiveness, or calibrated customer satisfaction. Rating,
  89 review count, novelty and mode merchandising do not raise this coverage score.
```


## `backend/app/data/vision/README.md:1–87`

```text
   1 # Local face and visible colour analysis
   2 
   3 This directory contains the **real, unquantized YuNet March 2023 face detector**
   4 used by `app.services.face_vision`. It runs with OpenCV DNN on the backend CPU.
   5 There is no external photo service, face recognition, identity matching, or
   6 medical classifier in this implementation.
   7 
   8 ## Model provenance and licence
   9 
  10 - Upstream: [OpenCV Zoo — YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  11 - Immutable source revision: `f12e12798e8314f7c074a6656816c048dcc95b7a`
  12 - [Exact model download](https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
  13 - File: `face_detection_yunet_2023mar.onnx`, 232,589 bytes
  14 - SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
  15 - Licence: MIT, Copyright (c) 2020 Shiqi Yu; the unmodified upstream licence
  16   is included as [LICENSE](LICENSE). The model-directory licence covers weights.
  17 - Runtime dependencies: `opencv-python-headless==4.13.0.92` and `numpy==2.2.6`;
  18   decoding and colour-profile conversion use the pinned Pillow dependency.
  19 
  20 The upstream detector reports a box, two eye centres, nose tip and two mouth
  21 corners. The colour measurement below is our deterministic post-processing;
  22 YuNet does **not** infer skin colour, skin health, undertone or Fitzpatrick type.
  23 
  24 ## Processing and quality decisions
  25 
  26 1. Decode bounded JPEG/PNG bytes (at most 5,000,000 bytes and 20 million pixels),
  27    apply EXIF orientation, and transform embedded ICC profiles to sRGB. Images
  28    without a profile are assumed sRGB. Invalid profiles and nonopaque images
  29    are rejected instead of measuring hidden or uninterpretable pixels.
  30 2. Resize to at most 640 pixels on the longest edge. Detect faces using YuNet,
  31    score threshold 0.85 and NMS threshold 0.3. A separate detector per request
  32    avoids sharing mutable OpenCV state across worker threads.
  33 3. Require exactly one sufficiently large, uncropped face near a frontal pose.
  34    Closed eyes do not prevent analysis: no iris or blink classifier is used.
  35 4. Reject excessive blur using a contrast-relative sharpness measure. Sample
  36    small circles on both cheeks, anchored by the eyes and mouth. Trim the
  37    brightest and darkest 15% of each patch's L* values, then use median sRGB.
  38 5. Reject clipped samples, strong side-to-side lighting differences,
  39    monochrome imagery and extreme blue/green or highly saturated colour casts.
  40    A low median pixel brightness alone **never** produces a quality rejection.
  41 6. Average the two robust cheek colours. Return a swatch and a provisional
  42    depth band from its measured CIE Lab L*. Undertone is always `unknown`.
  43 
  44 Quality issues are `no_face`, `multiple_faces`, `face_too_small`,
  45 `face_cropped`, `face_pose`, `blurred`, `exposure_clipped`, `uneven_lighting`,
  46 `color_unreliable`, `color_profile_invalid`, `invalid_image`, and
  47 `insufficient_skin_samples`. They return `status=retake`, without a depth or
  48 swatch. Missing, altered or unloadable weights are service-unavailable errors,
  49 not successful analysis or a photo-quality judgement.
  50 
  51 ## Provisional depth bands and limitations
  52 
  53 The five display bands are explicit product heuristics, **not calibrated
  54 measurements of a person's natural complexion**:
  55 
  56 | Measured CIE L* | Visible depth |
  57 | --- | --- |
  58 | 80 to 100 | fair |
  59 | 65 to below 80 | light |
  60 | 50 to below 65 | medium |
  61 | 35 to below 50 | tan |
  62 | below 35 | deep |
  63 
  64 These values describe the uploaded image in its lighting. They are **not
  65 Fitzpatrick phototypes**, sun-sensitivity scores, ethnicity, identity, or
  66 medical observations. Detector score is not presented as calibrated colour
  67 confidence. No undertone classifier is implemented.
  68 
  69 Illuminant colour, shadows, camera processing, makeup, filters and reflections
  70 can change the result. Symmetric lighting and two cheek samples cannot detect
  71 every uniform colour cast, cosmetic covering or subtle occlusion. There is no
  72 colour reference card, colour constancy calibration, learned skin segmentation,
  73 population-scale accuracy/fairness validation, or shade-matching guarantee.
  74 Do not automatically treat this provisional suggestion as a confirmed Beauty
  75 ID answer. Confirmation and correction belong to the consumer flow.
  76 
  77 ## Privacy, readiness and verification
  78 
  79 All decoding, detection and sampling happen in memory. The module does not
  80 write images, EXIF/ICC metadata, landmarks, face boxes, embeddings or crops to
  81 disk, analytics, logs or any remote provider. It returns only `FaceAnalysis`.
  82 The caller must enforce explicit photo consent and bounded worker capacity.
  83 
  84 `model_is_ready()` verifies the SHA-256 and loads the model without inference.
  85 Successful checks are cached by path and filesystem modification/change time
  86 and size; a replacement invalidates the cached check. Production deployments
  87 still require the project's privacy, validation and release gates.
```


## `backend/app/schemas.py:265–286`

```text
 265 SkinToneDepth = Literal["fair", "light", "medium", "tan", "deep"]
 266 CosmeticUndertone = Literal["unknown", "warm", "neutral", "cool"]
 267 
 268 
 269 class ComplexionProfile(BaseModel):
 270     """A user-confirmed cosmetic preference, never a biometric identity or diagnosis."""
 271 
 272     model_config = ConfigDict(extra="forbid")
 273     depth: SkinToneDepth
 274     undertone: CosmeticUndertone = "unknown"
 275     source: Literal["photo_confirmed", "manual"]
 276     scan_id: str | None = Field(default=None, min_length=1, max_length=120)
 277 
 278 
 279 class FaceAnalysis(BaseModel):
 280     status: Literal["ready", "retake"]
 281     depth: SkinToneDepth | None = None
 282     swatch_hex: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
 283     undertone: Literal["unknown"] = "unknown"
 284     face_count: int = Field(ge=0)
 285     issues: list[str] = Field(default_factory=list)
 286     model_version: str
```


## `backend/app/scan.py:112–156`

```text
 112 class LocalVisionScanProvider(ScanProvider):
 113     """CPU face model and measured visible color; no external calls or photo storage."""
 114 
 115     name = "local_yunet_complexion_v1"
 116     is_local = True
 117 
 118     def is_configured(self) -> bool:
 119         from .services.face_vision import model_is_ready
 120 
 121         return model_is_ready()
 122 
 123     async def analyze(self, photo: UploadedPhoto, beauty_id: BeautyID, resolved_mode: ResolvedMode) -> ScanResult:
 124         if photo.content is None:
 125             return await DevScanProvider().analyze(photo, beauty_id, resolved_mode)
 126         from .services.face_vision import FaceVisionUnavailable, analyze_face_photo
 127 
 128         try:
 129             analysis = await to_thread.run_sync(analyze_face_photo, photo.content, limiter=_vision_capacity)
 130         except FaceVisionUnavailable as exc:
 131             raise ProviderUnavailable(
 132                 "scan_vision_unavailable", "Face analysis is temporarily unavailable.",
 133                 kind=ProviderErrorKind.UNAVAILABLE,
 134             ) from exc
 135         scan_id = str(uuid.uuid4())
 136         ready = analysis.status == "ready" and analysis.depth is not None
 137         service = get_recommendation_service()
 138         shades = []
 139         if ready:
 140             catalog = service.catalog_index()
 141             recs, _ = service.recommend(
 142                 request=RecommendationsRequest(beauty_id=beauty_id, limit=10),
 143                 beauty_id=beauty_id, active_mode=resolved_mode.mode,
 144                 mode_confidence=resolved_mode.confidence, mode_source=resolved_mode.source,
 145                 catalog=catalog,
 146             )
 147             shades = service.shade_candidates(
 148                 beauty_id=beauty_id, depth=analysis.depth, undertone="unknown",
 149                 active_mode=resolved_mode.mode, mode_confidence=resolved_mode.confidence,
 150                 mode_source=resolved_mode.source, catalog=catalog,
 151             )
 152         else:
 153             recs = RecommendationsResponse(
 154                 explanation="Для определения тона нужен новый снимок.",
 155                 disclaimer="Подбор не изменён.", generated_at=datetime.now(timezone.utc),
 156             )
```


## `backend/app/routes/scan.py:20–34`

```text
  20 @router.post("/v1/photo/scan", response_model=ScanResult)
  21 async def scan(
  22     request: Request,
  23     account: Annotated[StoredAccount, Depends(current_account)],
  24     source: str = Form("questionnaire"),
  25     beauty_id_json: str | None = Form(None),
  26     photo: UploadFile | None = File(None),
  27     photo_consent: bool = Form(False),
  28 ) -> ScanResult:
  29     enforce_rate_limit(request, "scan", account_id=account.account_id)
  30     if not get_feature_flags().scan_enabled:
  31         raise HTTPException(status_code=403, detail="feature_disabled")
  32     store = get_store()
  33     if photo is not None and not photo_consent:
  34         raise HTTPException(status_code=400, detail="photo_consent_required")
```


## `backend/app/services/face_vision.py:145–156`

```text
 145 def _visible_depth(lightness: float) -> SkinToneDepth:
 146     # Provisional display bands of measured CIE L* under the photo's lighting.
 147     # These deliberately have no undertone or clinical interpretation.
 148     if lightness >= 80:
 149         return "fair"
 150     if lightness >= 65:
 151         return "light"
 152     if lightness >= 50:
 153         return "medium"
 154     if lightness >= 35:
 155         return "tan"
 156     return "deep"
```


## `backend/app/services/face_vision.py:264–277`

```text
 264     sampled_rgb = np.rint(cheeks.mean(axis=0)).astype(np.uint8)
 265     lightness = float(
 266         cv2.cvtColor(
 267             sampled_rgb[np.newaxis, np.newaxis].astype(np.float32) / 255,
 268             cv2.COLOR_RGB2LAB,
 269         )[0, 0, 0]
 270     )
 271     return FaceAnalysis(
 272         status="ready",
 273         face_count=1,
 274         depth=_visible_depth(lightness),
 275         swatch_hex="#" + "".join(f"{channel:02X}" for channel in sampled_rgb),
 276         model_version=MODEL_VERSION,
 277     )
```
