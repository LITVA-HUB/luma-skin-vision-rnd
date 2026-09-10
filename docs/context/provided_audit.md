# Luma: аудиторские свидетельства

Дата: 10 сентября 2026 года.

## Область проверки

Архив распакован без исполнения приложения. Сначала изучались структура, README, архитектура, PRD, CURRENT_STATUS, privacy, production readiness, алгоритмы, model card, зависимости, описания тестов. Затем адресно проверены vision, scan, схемы и профильные тесты. UI, CRUD и весь репозиторий подряд не анализировались.

Это не аудит безопасности всего приложения и не проверка точности измерения кожи. Среда ниже — среда анализа, НЕ компьютер пользователя с RTX 4060.

Архив: `Luma-2.0-main (1)(1).zip`

SHA-256 архива: `f0ce109f11b3eb5c5e026b2fdb3ae198fa6104c37464b5c6e63a2b517fc528cc`

YuNet: 232589 байт; SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.

## Независимый ограниченный запуск тестов

Команда:

```bash
env -u TEST_POSTGRES_URL -u TEST_REDIS_URL LOAD_LOCAL_ENV_FILE=false python -m pytest backend/tests/test_face_vision.py -q
```

Код возврата: 0. Вывод:

```text
.........................                                                [100%]
25 passed in 3.54s

```

Версии среды анализа (не все совпадают с requirements проекта):

- opencv-python-headless: 4.13.0.92
- numpy: 2.3.5
- pillow: 12.3.0
- pydantic: 2.13.4
- pytest: 9.0.2

Полный backend suite и iOS suite не запускались. Время pytest не является временем inference. Цветовая точность, ΔE00, risk–coverage, cross-device accuracy и показатели на RTX 4060 — НЕ ИЗМЕРЕНО.

## Фрагменты исходных материалов

Нумерация ниже соответствует строкам исходных файлов в переданном ZIP, а не строкам этого приложения. Факт наличия утверждения в документации не означает независимую проверку результата. Передача приватного ZIP не доказывает публичное раскрытие.

### `README.md`

Строки 19–26:

```text
19: что хочется оставить в ежедневном уходе.
20: 
21: **Нативный iPhone-клиент · SwiftUI · FastAPI · 240 демонстрационных SKU**
22: 
23: Локальный пилот для «Золотого Яблока». Каталог и сценарии на экранах —
24: демонстрационные; реальные магазинные интеграции ещё не подключены.
25: [Текущий статус →](docs/CURRENT_STATUS.md)
26: 
```

Строки 90–112:

```text
90: - Советник рекомендует известные доступные SKU текущего каталога. Карточки,
91:   цены и действия формирует backend; iOS не обращается к LLM напрямую.
92: - Медицинские запросы обрабатываются безопасным отказом без вызова LLM.
93:   Режимы не определяют диагнозы, гормоны или психологический тип.
94: - Сырые фото, координаты лица и полные данные аккаунта не отправляются в OpenRouter.
95:   Локальный анализ фото выполняется в памяти сервера; снимки не сохраняются.
96: - Демокаталог содержит `LUMA-001`…`LUMA-240`. Свойства, цены и изображения —
97:   тестовые данные проекта, а не реальный ассортимент магазина.
98: - Наборы ухода, полка, пополнение и сценарии повторного визита включаются через feature flags.
99: 
100: [Архитектура советника](docs/AI_AGENT_AND_BACKEND_ARCHITECTURE.md) ·
101: [Приватность](docs/SECURITY_PRIVACY.md) · [Feature flags](docs/feature-flags.md)
102: 
103: </details>
104: 
105: ## Попробовать сборку
106: 
107: **[Открыть предварительный релиз →](https://github.com/LITVA-HUB/Luma-2.0/releases/tag/pilot-2026.09.06)**
108: 
109: В релизе: **BeautyConcierge Staging для iOS / arm64**, параметры сборки,
110: SHA-256 и исходники. Архив содержит неподписанный `.app`: для установки
111: на iPhone нужна подпись Apple. Это не пакет TestFlight или App Store.
112: Сборка использует staging API; скриншоты выше сняты отдельно в Debug-демосценариях.
```

Строки 184–190:

```text
184: 
185: **[Статус](docs/CURRENT_STATUS.md)** · **[QA](docs/QA_REPORT.md)** ·
186: **[Критерии релиза](docs/PRODUCTION_RELEASE_CRITERIA.md)** · **[Live-пилот](docs/LIVE_PILOT.md)**
187: 
188: ---
189: 
190: <p align="center"><strong>ЗОЛОТОЕ ЯБЛОКО / BEAUTY ID</strong><br /><sub>Internal · Proprietary · Права на бренд и ассеты требуют отдельной проверки перед внешним распространением.</sub></p>
```

### `backend/app/data/vision/README.md`

Строки 1–99:

```text
1: # Local face and visible colour analysis
2: 
3: This directory contains the **real, unquantized YuNet March 2023 face detector**
4: used by `app.services.face_vision`. It runs with OpenCV DNN on the backend CPU.
5: There is no external photo service, face recognition, identity matching, or
6: medical classifier in this implementation.
7: 
8: ## Model provenance and licence
9: 
10: - Upstream: [OpenCV Zoo — YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
11: - Immutable source revision: `f12e12798e8314f7c074a6656816c048dcc95b7a`
12: - [Exact model download](https://media.githubusercontent.com/media/opencv/opencv_zoo/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
13: - File: `face_detection_yunet_2023mar.onnx`, 232,589 bytes
14: - SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
15: - Licence: MIT, Copyright (c) 2020 Shiqi Yu; the unmodified upstream licence
16:   is included as [LICENSE](LICENSE). The model-directory licence covers weights.
17: - Runtime dependencies: `opencv-python-headless==4.13.0.92` and `numpy==2.2.6`;
18:   decoding and colour-profile conversion use the pinned Pillow dependency.
19: 
20: The upstream detector reports a box, two eye centres, nose tip and two mouth
21: corners. The colour measurement below is our deterministic post-processing;
22: YuNet does **not** infer skin colour, skin health, undertone or Fitzpatrick type.
23: 
24: ## Processing and quality decisions
25: 
26: 1. Decode bounded JPEG/PNG bytes (at most 5,000,000 bytes and 20 million pixels),
27:    apply EXIF orientation, and transform embedded ICC profiles to sRGB. Images
28:    without a profile are assumed sRGB. Invalid profiles and nonopaque images
29:    are rejected instead of measuring hidden or uninterpretable pixels.
30: 2. Resize to at most 640 pixels on the longest edge. Detect faces using YuNet,
31:    score threshold 0.85 and NMS threshold 0.3. A separate detector per request
32:    avoids sharing mutable OpenCV state across worker threads.
33: 3. Require exactly one sufficiently large, uncropped face near a frontal pose.
34:    Closed eyes do not prevent analysis: no iris or blink classifier is used.
35: 4. Reject excessive blur using a contrast-relative sharpness measure. Sample
36:    small circles on both cheeks, anchored by the eyes and mouth. Trim the
37:    brightest and darkest 15% of each patch's L* values, then use median sRGB.
38: 5. Reject clipped samples, strong side-to-side lighting differences,
39:    monochrome imagery and extreme blue/green or highly saturated colour casts.
40:    A low median pixel brightness alone **never** produces a quality rejection.
41: 6. Average the two robust cheek colours. Return a swatch and a provisional
42:    depth band from its measured CIE Lab L*. Undertone is always `unknown`.
43: 
44: Quality issues are `no_face`, `multiple_faces`, `face_too_small`,
45: `face_cropped`, `face_pose`, `blurred`, `exposure_clipped`, `uneven_lighting`,
46: `color_unreliable`, `color_profile_invalid`, `invalid_image`, and
47: `insufficient_skin_samples`. They return `status=retake`, without a depth or
48: swatch. Missing, altered or unloadable weights are service-unavailable errors,
49: not successful analysis or a photo-quality judgement.
50: 
51: ## Provisional depth bands and limitations
52: 
53: The five display bands are explicit product heuristics, **not calibrated
54: measurements of a person's natural complexion**:
55: 
56: | Measured CIE L* | Visible depth |
57: | --- | --- |
58: | 80 to 100 | fair |
59: | 65 to below 80 | light |
60: | 50 to below 65 | medium |
61: | 35 to below 50 | tan |
62: | below 35 | deep |
63: 
64: These values describe the uploaded image in its lighting. They are **not
65: Fitzpatrick phototypes**, sun-sensitivity scores, ethnicity, identity, or
66: medical observations. Detector score is not presented as calibrated colour
67: confidence. No undertone classifier is implemented.
68: 
69: Illuminant colour, shadows, camera processing, makeup, filters and reflections
70: can change the result. Symmetric lighting and two cheek samples cannot detect
71: every uniform colour cast, cosmetic covering or subtle occlusion. There is no
72: colour reference card, colour constancy calibration, learned skin segmentation,
73: population-scale accuracy/fairness validation, or shade-matching guarantee.
74: Do not automatically treat this provisional suggestion as a confirmed Beauty
75: ID answer. Confirmation and correction belong to the consumer flow.
76: 
77: ## Privacy, readiness and verification
78: 
79: All decoding, detection and sampling happen in memory. The module does not
80: write images, EXIF/ICC metadata, landmarks, face boxes, embeddings or crops to
81: disk, analytics, logs or any remote provider. It returns only `FaceAnalysis`.
82: The caller must enforce explicit photo consent and bounded worker capacity.
83: 
84: `model_is_ready()` verifies the SHA-256 and loads the model without inference.
85: Successful checks are cached by path and filesystem modification/change time
86: and size; a replacement invalidates the cached check. Production deployments
87: still require the project's privacy, validation and release gates.
88: 
89: Run from the repository root:
90: 
91: ```bash
92: PYTHONPATH=backend .venv/bin/python -m pytest backend/tests/test_face_vision.py -q
93: ```
94: 
95: Tests exercise the real bundled model against the repository's existing
96: `scan_test_photo.jpg`, real two-face montage, small/darkened/blurred/monochrome
97: variants, EXIF/ICC handling, model failures and concurrent inference. Controlled
98: landmarks isolate pose, crop, clipping and uneven-light post-processing gates.
99: They verify mechanics and failure behaviour, not real-world colour accuracy.
```

### `backend/app/services/face_vision.py`

Строки 145–156:

```text
145: def _visible_depth(lightness: float) -> SkinToneDepth:
146:     # Provisional display bands of measured CIE L* under the photo's lighting.
147:     # These deliberately have no undertone or clinical interpretation.
148:     if lightness >= 80:
149:         return "fair"
150:     if lightness >= 65:
151:         return "light"
152:     if lightness >= 50:
153:         return "medium"
154:     if lightness >= 35:
155:         return "tan"
156:     return "deep"
```

Строки 202–277:

```text
202:     gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
203:     crop = gray[int(y) : int(y + box_height), int(x) : int(x + box_width)]
204:     crop = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_AREA)
205:     # A contrast-relative sharpness test avoids treating darker pixels as blur.
206:     sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) / max(
207:         float(crop.var()), 1.0
208:     )
209:     if sharpness < 0.012:
210:         return _retake(1, "blurred")
211: 
212:     radius = eye_distance * 0.105
213:     yy, xx = np.ogrid[:height, :width]
214:     cheek_samples = []
215:     for eye, inward in ((left_eye, 1), (right_eye, -1)):
216:         center = (
217:             eye
218:             + inward * eye_distance * 0.06 * horizontal
219:             + eye_to_mouth * 0.60 * vertical
220:         )
221:         if (
222:             center[0] - radius < 0
223:             or center[0] + radius >= width
224:             or center[1] - radius < 0
225:             or center[1] + radius >= height
226:         ):
227:             return _retake(1, "face_cropped")
228:         mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
229:         samples = rgb[mask]
230:         if len(samples) < 64:
231:             return _retake(1, "insufficient_skin_samples")
232:         # Reject lost pixel information, never a low median skin brightness.
233:         clipped = np.all(samples <= 3, axis=1) | np.any(samples >= 252, axis=1)
234:         if float(clipped.mean()) > 0.10:
235:             return _retake(1, "exposure_clipped")
236:         samples = samples[~clipped]
237:         lab = cv2.cvtColor(
238:             samples[np.newaxis].astype(np.float32) / 255, cv2.COLOR_RGB2LAB
239:         )[0]
240:         # Trim bright highlights and small darker spots before taking a median.
241:         low, high = np.quantile(lab[:, 0], (0.15, 0.85))
242:         robust = samples[(lab[:, 0] >= low) & (lab[:, 0] <= high)]
243:         if len(robust) < 32:
244:             return _retake(1, "insufficient_skin_samples")
245:         cheek_samples.append(np.median(robust, axis=0))
246: 
247:     cheeks = np.array(cheek_samples, dtype=np.float32)
248:     cheek_lab = cv2.cvtColor(cheeks[np.newaxis] / 255, cv2.COLOR_RGB2LAB)[0]
249:     # Monochrome pixels and strong green/blue casts do not support a colour
250:     # estimate. This is not an undertone classifier or a skin-brightness gate.
251:     monochrome_fraction = float((np.ptp(rgb, axis=2) <= 2).mean())
252:     if (
253:         monochrome_fraction > 0.98
254:         or np.any(cheek_lab[:, 1:] < -8)
255:         or np.any(np.linalg.norm(cheek_lab[:, 1:], axis=1) > 65)
256:     ):
257:         return _retake(1, "color_unreliable")
258:     # Large side-to-side colour differences are lighting/occlusion ambiguity.
259:     if (
260:         abs(float(cheek_lab[0, 0] - cheek_lab[1, 0])) > 12
261:         or float(np.linalg.norm(cheek_lab[0, 1:] - cheek_lab[1, 1:])) > 15
262:     ):
263:         return _retake(1, "uneven_lighting")
264:     sampled_rgb = np.rint(cheeks.mean(axis=0)).astype(np.uint8)
265:     lightness = float(
266:         cv2.cvtColor(
267:             sampled_rgb[np.newaxis, np.newaxis].astype(np.float32) / 255,
268:             cv2.COLOR_RGB2LAB,
269:         )[0, 0, 0]
270:     )
271:     return FaceAnalysis(
272:         status="ready",
273:         face_count=1,
274:         depth=_visible_depth(lightness),
275:         swatch_hex="#" + "".join(f"{channel:02X}" for channel in sampled_rgb),
276:         model_version=MODEL_VERSION,
277:     )
```

### `backend/app/scan.py`

Строки 109–176:

```text
109: _vision_capacity = CapacityLimiter(2)
110: 
111: 
112: class LocalVisionScanProvider(ScanProvider):
113:     """CPU face model and measured visible color; no external calls or photo storage."""
114: 
115:     name = "local_yunet_complexion_v1"
116:     is_local = True
117: 
118:     def is_configured(self) -> bool:
119:         from .services.face_vision import model_is_ready
120: 
121:         return model_is_ready()
122: 
123:     async def analyze(self, photo: UploadedPhoto, beauty_id: BeautyID, resolved_mode: ResolvedMode) -> ScanResult:
124:         if photo.content is None:
125:             return await DevScanProvider().analyze(photo, beauty_id, resolved_mode)
126:         from .services.face_vision import FaceVisionUnavailable, analyze_face_photo
127: 
128:         try:
129:             analysis = await to_thread.run_sync(analyze_face_photo, photo.content, limiter=_vision_capacity)
130:         except FaceVisionUnavailable as exc:
131:             raise ProviderUnavailable(
132:                 "scan_vision_unavailable", "Face analysis is temporarily unavailable.",
133:                 kind=ProviderErrorKind.UNAVAILABLE,
134:             ) from exc
135:         scan_id = str(uuid.uuid4())
136:         ready = analysis.status == "ready" and analysis.depth is not None
137:         service = get_recommendation_service()
138:         shades = []
139:         if ready:
140:             catalog = service.catalog_index()
141:             recs, _ = service.recommend(
142:                 request=RecommendationsRequest(beauty_id=beauty_id, limit=10),
143:                 beauty_id=beauty_id, active_mode=resolved_mode.mode,
144:                 mode_confidence=resolved_mode.confidence, mode_source=resolved_mode.source,
145:                 catalog=catalog,
146:             )
147:             shades = service.shade_candidates(
148:                 beauty_id=beauty_id, depth=analysis.depth, undertone="unknown",
149:                 active_mode=resolved_mode.mode, mode_confidence=resolved_mode.confidence,
150:                 mode_source=resolved_mode.source, catalog=catalog,
151:             )
152:         else:
153:             recs = RecommendationsResponse(
154:                 explanation="Для определения тона нужен новый снимок.",
155:                 disclaimer="Подбор не изменён.", generated_at=datetime.now(timezone.utc),
156:             )
157:         labels = {"fair": "очень светлый", "light": "светлый", "medium": "средний", "tan": "смуглый", "deep": "глубокий"}
158:         return ScanResult(
159:             scan_id=scan_id, image_analyzed=True, analysis=analysis, shade_candidates=shades,
160:             summary=(f"Видимый тон на снимке — {labels[analysis.depth]}. Подтвердите или поправьте результат." if ready
161:                      else "Не удалось надёжно оценить тон. Переснимите лицо при ровном дневном свете."),
162:             signals=(["Найдено одно лицо", "Цвет измерен на участках щёк", "Подтон можно указать вручную"] if ready else []),
163:             limitations=[
164:                 "Освещение, макияж и камера влияют на видимый цвет; это ориентир для примерки, не точный номер оттенка.",
165:                 "Подтон по некалиброванному фото не определяется. Диагностика кожи не выполняется.",
166:                 "Фото и координаты лица обрабатываются в памяти нашего сервера и не сохраняются.",
167:             ],
168:             statuses=[
169:                 ScanStatus(key="preparing", label="Подготовка", is_done=True),
170:                 ScanStatus(key="uploading", label="Загрузка", is_done=True),
171:                 ScanStatus(key="analyzing", label="Лицо и видимый тон", is_done=True),
172:                 ScanStatus(key="ready" if ready else "failed", label="Подтвердите тон" if ready else "Нужен новый снимок", is_done=ready),
173:             ],
174:             recommendations=recs,
175:             retention_policy="Raw photos, face landmarks and measured colors are not stored. Only the cosmetic tone you confirm is saved in Beauty ID.",
176:             deletion_url=f"/v1/photo/scan/{scan_id}",
```

### `backend/app/schemas.py`

Строки 265–287:

```text
265: SkinToneDepth = Literal["fair", "light", "medium", "tan", "deep"]
266: CosmeticUndertone = Literal["unknown", "warm", "neutral", "cool"]
267: 
268: 
269: class ComplexionProfile(BaseModel):
270:     """A user-confirmed cosmetic preference, never a biometric identity or diagnosis."""
271: 
272:     model_config = ConfigDict(extra="forbid")
273:     depth: SkinToneDepth
274:     undertone: CosmeticUndertone = "unknown"
275:     source: Literal["photo_confirmed", "manual"]
276:     scan_id: str | None = Field(default=None, min_length=1, max_length=120)
277: 
278: 
279: class FaceAnalysis(BaseModel):
280:     status: Literal["ready", "retake"]
281:     depth: SkinToneDepth | None = None
282:     swatch_hex: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
283:     undertone: Literal["unknown"] = "unknown"
284:     face_count: int = Field(ge=0)
285:     issues: list[str] = Field(default_factory=list)
286:     model_version: str
287: 
```

### `backend/tests/test_face_vision.py`

Строки 13–16:

```text
13: PHOTO_PATH = (
14:     Path(__file__).resolve().parents[2]
15:     / "ios/BeautyConcierge/Resources/Assets.xcassets/scan_test_photo.imageset/scan_test_photo.jpg"
16: )
```

Строки 30–49:

```text
30: def test_real_yunet_inference_returns_visible_color_without_identity_or_medical_labels():
31:     from app.services.face_vision import analyze_face_photo, model_is_ready
32: 
33:     assert model_is_ready()
34:     result = analyze_face_photo(PHOTO_PATH.read_bytes())
35:     assert result.status == "ready"
36:     assert result.face_count == 1
37:     assert result.depth == "light"
38:     assert result.swatch_hex is not None
39:     assert result.undertone == "unknown"
40:     assert result.issues == []
41:     assert set(result.model_dump()) == {
42:         "status",
43:         "depth",
44:         "swatch_hex",
45:         "undertone",
46:         "face_count",
47:         "issues",
48:         "model_version",
49:     }
```

Строки 87–114:

```text
87: def test_dark_visible_skin_samples_are_not_rejected_as_insufficient_light():
88:     from app.services.face_vision import analyze_face_photo
89: 
90:     dark = Image.fromarray((np.array(_portrait()) * 0.25).astype(np.uint8))
91:     result = analyze_face_photo(_encode(dark))
92:     assert result.status == "ready"
93:     assert result.depth == "deep"
94:     assert result.swatch_hex is not None
95:     assert result.issues == []
96: 
97: 
98: @pytest.mark.parametrize("transformation", ["grayscale", "blue_cast"])
99: def test_photograph_without_reliable_skin_color_cannot_produce_depth(transformation):
100:     from app.services.face_vision import analyze_face_photo
101: 
102:     portrait = _portrait()
103:     if transformation == "grayscale":
104:         portrait = ImageOps.grayscale(portrait).convert("RGB")
105:     else:
106:         pixels = np.array(portrait).astype(np.float32)
107:         pixels *= np.array([0.55, 0.8, 1.2])
108:         portrait = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
109:     result = analyze_face_photo(_encode(portrait))
110:     assert result.face_count == 1
111:     assert result.status == "retake"
112:     assert "color_unreliable" in result.issues
113:     assert result.depth is None
114:     assert result.swatch_hex is None
```

### `docs/ARCHITECTURE.md`

Строки 96–120:

```text
96: ## Reference recommendation core
97: 
98: All automated product decisions use one deterministic provider-independent
99: pipeline:
100: 
101: ```text
102: CatalogProvider.get_snapshot() (snapshot-owned deeply immutable Products)
103:   -> CatalogIndex.build() (complete fail-closed validation + normalized facts)
104:   -> RecommendationContext (Beauty ID + bounded focus + supported filters
105:                              + authoritative mode + seen SKU history)
106:   -> EligibilityPolicy (hard rejections)
107:   -> RecommendationRanker (typed lexicographic FitVector)
108:   -> RecommendationEngine (limited, grounded decision)
109:   -> RecommendationPresenter (existing public DTO)
110: ```
111: 
112: The pure engine receives only a context and catalog index. It does not read the
113: database, provider, environment, wall clock, HTTP, FastAPI, or an LLM. Identical
114: context, catalog version, and `reference-v3` engine semantics therefore produce
115: the same canonical-SKU order for structurally valid unique-SKU snapshots,
116: regardless of provider response order or SKU prefix. Malformed duplicate-SKU
117: snapshots fail closed before ranking. Routine builder, deterministic
118: advisor compiled tasks, scan fallback,
119: win-back, and PDP rationale reuse the same eligibility/ranking boundary instead
120: of defining independent recommendation policies.
```

Строки 134–146:

```text
134: ### Historical reference-v1 benchmark and provisional pilot SLO
135: 
136: These measurements predate `reference-v3`. They document an earlier baseline,
137: not current latency or a verified capacity guarantee for this revision.
138: 
139: `scripts/benchmark_reference_catalog.py` is a deterministic, no-network
140: benchmark that starts a fresh process for every size and exercises generated
141: `Product` records through production `CatalogSnapshot`, `CatalogIndex`, and
142: warm `RecommendationService` paths. Generation, immutable snapshot/content-SHA
143: construction, first index build, and warm decision timings are measured
144: separately. RSS is normalized to bytes (process peak on macOS, current RSS on
145: Linux). The lightweight pytest contract checks output shape and snapshot/index
146: reuse with a tiny catalog; normal pytest has no hardware-sensitive timing gate.
```

Строки 221–243:

```text
221: ## Development catalog
222: 
223: `backend/app/data/catalog.json` contains 240 synthetic Luma products. The first
224: 94 records came from the original asset bundle; 146 demo variants extend
225: coverage without inventing missing metadata on those original products.
226: `scripts/build_demo_catalog_extension.py --check` verifies the generated catalog.
227: 
228: The current seed happens to use unique backend SKUs `LUMA-001` through
229: `LUMA-240`; this is fixture data, not an algorithmic convention. Runtime
230: identity is the configured provider's canonical `Product.sku`. The non-unique
231: label from the seed card is stored as `source_sku` only so duplicate source
232: labels do not collide in cart, product detail, recommendations or advisor
233: grounding.
234: 
235: Product cards are copied to `backend/app/static/assets/cards/` and served by FastAPI from `/assets/cards/`. This catalog is still development/staging synthetic data; production must connect the external licensed catalog/CDN.
236: 
237: A real Golden Apple integration implements `GoldenAppleCatalogProvider`, maps
238: each raw retail record to the existing `Product` contract without inferring
239: missing attributes, conditionally validates an upstream revision/ETag, and
240: publishes through `AtomicCachedCatalogProvider`. The upstream token is separate
241: from the required SHA-256 content version. Eligibility, ranking, engine,
242: routine, and advisor modules do not change for that integration. The production
243: adapter is not implemented today.
```

### `docs/AI_AGENT_AND_BACKEND_ARCHITECTURE.md`

Строки 94–119:

```text
94: ## Советник: от текста до карточек
95: 
96: Вход — `POST /v1/advisor/message`, `routes/advisor.py`.
97: 
98: 1. Preflight в `advisor_safety.py` проверяет медицинский риск. Запрос на
99:    диагноз/лечение получает безопасный ответ без вызова LLM.
100: 2. Роут читает Beauty ID, режим, историю текущей сессии, корзину и набор.
101:    `advisor_references.py` формирует серверные ссылки на допустимый контекст.
102:    Произвольное клиентское `current_skus` не выбирает полномочия.
103: 3. `advisor_privacy.py` создаёт ограниченный `AdvisorProviderInput`: очищенное
104:    сообщение, очищенные предыдущие реплики, сводку ссылок и версию онтологии.
105:    Профиль, реальные account/session IDs, цены и полные карточки не отправляются
106:    как контекст. Явно написанный пользователем бюджет сохраняется для разбора
107:    ограничения. Приватные значения и служебные маркеры очищаются.
108:    Фото в этом вызове не участвует.
109: 4. Провайдер возвращает `AdvisorIntent` из `advisor_intent.py` /
110:    `advisor_contracts.py`. Схема и семантическая проверка различают разговор,
111:    уточнение, подбор и предложение изменения. Корректный JSON ещё не даёт
112:    разрешения выполнить содержащуюся в нём задачу.
113: 5. `advisor_compiler.py` связывает намерение с запросом, бюджетом,
114:    ограничениями и серверными ссылками. Пропуск распознанного текущего потолка
115:    в поддерживаемой денежной записи ведёт к уточнению без товаров; это
116:    ограниченный разбор, не гарантия понимания любой формулировки бюджета.
117:    Неоднозначная/устаревшая ссылка требует уточнения без изменения списков.
118: 6. `advisor_projection.py` исполняет задачу через общий подбор, формирует
119:    receipt и публичный ответ. Postflight сверяет карточки и текст с receipt
```

Строки 188–195:

```text
188: ## Остальные сценарии
189: 
190: - Фото: `routes/scan.py` → `scan.py` → `services/face_vision.py`.
191:   Проверяются согласие, размер и формат. YuNet ищет лицо, измерение цвета
192:   предлагает видимую глубину тона. Освещение и ракурс влияют на результат.
193:   Тип кожи, диагноз и гормоны по фото не определяются. Сохраняются
194:   подтверждённое предпочтение и ограниченная история, без фото, landmarks
195:   и исходных цветовых образцов.
```

### `docs/API_CONTRACTS.md`

Строки 332–378:

```text
332: ## Photo / scan
333: 
334: - `POST /photo/scan` multipart: `source`, optional `beauty_id_json`, optional
335:   `photo`, `photo_consent` (false by default). A file requires explicit photo
336:   consent or returns `400 photo_consent_required`; questionnaire consent is separate.
337: - Invalid multipart profile JSON/schema returns `422 invalid_beauty_id_json`.
338: - JPEG/PNG require matching MIME, complete decoding, at most 5 MB / 20 MP.
339:   Empty/truncated/false-format files return `415`, excessive sizes `413`.
340:   iOS normalizes library images (including HEIC/HEIF) to JPEG before upload.
341: - `SCAN_PROVIDER=local_vision` (default) uses a bundled, checksummed YuNet face
342:   detector and deterministic visible-color measurement on the backend CPU.
343:   Processing stays in memory, with at most two concurrent inference workers.
344:   No photo/landmarks/color samples are stored or sent to an external provider.
345: - Successful processing sets `image_analyzed=true` and returns `analysis`:
346:   `{status: ready|retake, depth: fair|light|medium|tan|deep|null,
347:   swatch_hex: #RRGGBB|null, undertone: unknown, face_count, issues, model_version}`.
348:   `retake` is a completed quality check without a reliable tone, not a matched
349:   profile. It carries no shade candidates or product recommendations.
350: - `ready` provides an editable cosmetic estimate and `shade_candidates` from
351:   exact catalog evidence. These are candidates for trying, not a guaranteed
352:   shade match. It does not mutate Beauty ID, active selection or cart.
353: - Missing/corrupt model/runtime fails explicitly (`503 scan_vision_unavailable`);
354:   it never falls back to claimed analysis. `/ready` exposes model availability.
355: - No-file requests keep the questionnaire-only path (`image_analyzed=false`,
356:   `analysis=null`). Explicit `SCAN_PROVIDER=dev` is a questionnaire-only test
357:   adapter. `external` remains an unimplemented provider contract.
358: 
359: The client confirms/corrects the estimate with `PATCH /beauty-id`:
360: `{"complexion":{"depth":"light","undertone":"unknown","source":"photo_confirmed","scan_id":"..."}}`.
361: `source` is `photo_confirmed` or `manual`; a known undertone is a user preference,
362: not inferred by the photo estimator. PATCH requires an existing consenting profile.
363: Pre-questionnaire clients may include the confirmed preference in the first
364: consented `PUT /beauty-id`. Only this minimal preference is persisted in the
365: existing profile JSON; no database migration is needed.
366: 
367: Omitting `complexion` preserves it on PUT/PATCH; explicit null removes it.
368: `GET /beauty-id` and privacy export include it. Confirmed depth can softly rank
369: powders and populate the separate `RecommendationsResponse.shade_candidates`
370: block for broad/makeup requests. Skincare preferences, ingredient/fragrance
371: constraints and the exact-variant gate remain in effect. Missing shade metadata
372: or unsupported depth/undertone yields an empty candidate list.
373: 
374: `DELETE /photo/scan/{scan_id}` atomically clears only the matching account's
375: confirmed complexion from that scan and records the privacy request. Raw photo,
376: measured color and landmark deletion is unnecessary for this local adapter
377: because none are retained. Account deletion removes the preference with the
378: profile. The photo step is optional; skipping never fabricates consent.
```

### `docs/SECURITY_PRIVACY.md`

Строки 94–113:

```text
94: ## Photo lifecycle
95: 
96: - The photo feature is optional and needs an explicit action plus separate photo
97:   consent. The questionnaire remains usable without a photo.
98: - JPEG/PNG MIME, full decoding, compressed size and expanded pixels are validated.
99:   The iOS client normalizes HEIC/HEIF to JPEG and removes image metadata on upload.
100: - `local_vision` runs a bundled CPU face detector and measures visible cheek color
101:   in backend memory. It never sends photos to OpenRouter or another service.
102:   It does not recognize identity or infer ethnicity, health, hormones or skin type.
103: - Only an explicitly confirmed cosmetic depth, optional manually selected
104:   undertone, source and scan ID enter the account's Beauty ID JSON. The photo,
105:   landmarks and measured swatch are transient; crash events redact complexion,
106:   swatches and landmarks. The advisor uses confirmed preferences inside the
107:   backend; they are not serialized into its external provider input.
108: - `PATCH /v1/beauty-id` with `complexion:null` removes the preference;
109:   `DELETE /v1/photo/scan/{scan_id}` clears only the matching account's saved tone.
110:   Export includes confirmed complexion, and account deletion removes it.
111: - `dev` explicitly reports no image analysis. The optional external scan contract
112:   is still unimplemented. Operating this feature in production still requires
113:   representative quality evaluation and documented infrastructure/privacy policy.
```

### `docs/PRODUCTION_GAPS.md`

Строки 25–55:

```text
25: - Vendor-neutral direct/`NAME_FILE` secret resolution with strict file guards and
26:   value-based log/Sentry redaction.
27: - Unified provider capability/error/deadline/request-ID contracts, explicit
28:   local implementations, and honest external placeholders.
29: - Isolated PostgreSQL custom-format backup/restore drill with SHA-256,
30:   migration/sentinel/link verification, negative guards, and scoped cleanup.
31: - Deterministic `reference-v1` recommendation core with fail-closed catalog
32:   validation and hard eligibility, provider-order/SKU-prefix independence,
33:   shared recommendation/routine/advisor/scan/win-back/PDP policy, exact routine
34:   budgets, executable golden cases, generated invariants, and intentional
35:   abstention when verified metadata is insufficient.
36: - Atomic production catalog snapshot contract with conditional upstream token
37:   validation, immutable last-good publication, and request-path snapshot/index
38:   reuse. This is a provider boundary only; no Golden Apple adapter is connected.
39: 
40: ## Remaining before real TestFlight/App Store ❌
41: 
42: 1. Connect production auth provider (currently dev-login + guest + local tokens).
43: 2. Implement `GoldenAppleCatalogProvider`, raw retail record to `Product`
44:    mapping, conditional upstream revision/ETag validation, and licensed product
45:    image CDN integration (the current 94-SKU set is synthetic development
46:    data). The vendor token must populate `upstream_revision` separately from the
47:    required lowercase SHA-256 content version. The recommendation engine,
48:    eligibility, ranking, routine, and advisor policy must remain unchanged when
49:    this adapter is connected.
50: 3. Keep checkout intentionally disabled or connect a production checkout provider.
51: 4. Validate local YuNet/visible-tone analysis on representative photos and production infrastructure; approve retention/deletion policy. An external scan adapter remains optional and unimplemented.
52: 5. Connect SMS provider (phone auth) and push notifications.
53: 6. Deliver and rotate real OpenRouter, Sentry, Redis/database, provider, and admin
54:    credentials through protected production files or another compatible secret
55:    delivery mechanism; the repository includes resolution/redaction, not a hosted
```

Строки 80–86:

```text
80: result; production integration must improve verified metadata rather than weaken
81: the abstention rules.
82: 
83: The project may be described as public pilot-ready only after the `docs/LIVE_PILOT.md` checklist is complete. It must not be described as App Store-ready until the remaining provider, QA, legal/privacy, and backup items are complete.
84: 
85: ---
86: _Last updated: 2026-08-29 (atomic reference catalog contract documented;
```

### `docs/CURRENT_STATUS.md`

Строки 1–10:

```text
1: # Current Project Status
2: 
3: ## Повторная проверка приложения — 2026-09-08
4: 
5: Свежие проверки перед отправкой изменений в GitHub прошли:
6: **6827 backend passed / 10 skipped** на PostgreSQL/Redis,
7: **8 deterministic llm_eval passed**, **395 iOS unit passed** и
8: **62 iOS UI passed / 0 failed / 0 skipped** на подписанной сборке Simulator.
9: Дополнительно прошли 75 HTTP-проверок локального API, 16 HTTP-проверок
10: Linux/amd64 Docker-образа и сборки iOS Release/Staging без установки.
```

Строки 554–565:

```text
554: ## Git metadata
555: 
556: The workspace now has a real `main` branch and a configured private GitHub
557: remote at `LITVA-HUB/Luma-2.0`; `main` is the verified default branch. Earlier
558: review notes that describe a snapshot without `HEAD` or remote metadata are
559: historical boundaries from before the repository was published.
560: 
561: ## Product Position
562: 
563: The app is a Golden Apple branded pilot / release-candidate prototype. It is not production-ready until real retail providers, production infrastructure, scheduled offsite backups and provider-approved recovery rehearsal, managed secret delivery/rotation, legal surfaces, push, and the physical QA matrix are completed.
564: 
565: The visible iOS app name is `Золотое Яблоко`. Historical docs may still mention `Luma Beauty ID` where they describe earlier project phases or internal package names.
```

### `backend/requirements.txt`

Строки 1–14:

```text
1: fastapi==0.139.0
2: uvicorn==0.35.0
3: httpx==0.28.1
4: pydantic==2.11.7
5: python-multipart==0.0.32
6: Pillow==12.3.0
7: opencv-python-headless==4.13.0.92
8: numpy==2.2.6
9: psycopg[binary]==3.2.3
10: alembic==1.13.2
11: pytest==9.1.1
12: sentry-sdk[fastapi]>=2.0
13: pyyaml>=6.0.2
14: redis==7.4.1
```

