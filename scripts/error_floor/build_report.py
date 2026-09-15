"""Render measured aggregate audit results; fail closed on incomplete capture data."""
import argparse
import base64
import html
import json
from pathlib import Path


def f(x, n=3):
    return '—' if x is None else f'{x:.{n}f}'


class Document:
    def __init__(self):
        self.md, self.web = [], []

    def title(self, text, level=2):
        self.md.append('#'*level+' '+text+'\n')
        self.web.append(f'<h{level}>{html.escape(text)}</h{level}>')

    def paragraph(self, text):
        self.md.append(text+'\n')
        self.web.append('<p>'+html.escape(text)+'</p>')

    def table(self, columns, rows):
        rows = [[str(x) for x in row] for row in rows]
        self.md.append('| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+
                       '\n'.join('| '+' | '.join(row)+' |' for row in rows)+'\n')
        self.web.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+
                        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+
                        '</tbody></table></div>')

    def code(self, text):
        self.md.append('```bash\n'+text+'\n```\n')
        self.web.append('<pre>'+html.escape(text)+'</pre>')

    def figure(self, path):
        if path.exists():
            self.md.append('![Диагностические распределения](figures/'+path.name+')\n')
            self.web.append('<img alt="Диагностические распределения" src="data:image/png;base64,'+
                            base64.b64encode(path.read_bytes()).decode()+'">')

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'REPORT_RU.md').write_text('\n'.join(self.md))
        style = 'body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#18232c}h1{font-size:32px}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #d4dce1}th{background:#edf3f6}.table{overflow:auto}pre{padding:16px;background:#edf3f6;overflow:auto;font-size:13px}img{width:100%;height:auto}a{color:#175f89}@media print{body{margin:0;font-size:11px}h2{break-after:avoid}tr{break-inside:avoid}}'
        (folder/'REPORT_RU.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Luma · MSKCC error-floor audit</title><style>'+style+'</style><body>'+''.join(self.web)+'</body></html>')


def main(root, output):
    instrument = json.loads((root/'instrument.json').read_text())
    capture = json.loads((root/'capture/capture.json').read_text())
    cf = json.loads((root/'reference_capture_counterfactuals.json').read_text())
    if capture['scope'] != 'COMPLETE_TRAIN_CAPTURE_DIAGNOSTIC_NO_TRAINING':
        raise ValueError('Cannot publish partial-image audit as complete')
    i = instrument['instrument_repeatability']
    d = Document()
    d.title('Luma — MSKCC: diagnostic error-floor audit', 1)
    d.paragraph('Дата: 15 сентября 2026. Статус: измерения reference/capture завершены; декомпозиция ошибок C заблокирована отсутствием сохранённых OOF predictions. Новых моделей, quality predictor и обучения нет. Приведённые ниже результаты — новый диагностический расчёт на восстановленных исходных данных, не новый датасет и не проверка селфи.')
    d.table(['Диагностика ΔE00', 'n', 'mean', 'median', 'p90', 'p95', 'max'], [
        [name, s['n'], *[f(s[k]) for k in ['mean','median','p90','p95','max']]] for name,s in [
            ('Прибор: пары записанных assessments', i['pairwise_delta_e00']),
            ('Один assessment против среднего двух других', i['held_assessment_vs_other_two_mean']['delta_e00']),
            ('Изменение среднего target при исключении одного assessment', i['triplet_target_sensitivity_to_omitting_one_assessment']['delta_e00'])]])
    d.paragraph('Эти три величины отвечают на разные вопросы. Ни pairwise disagreement отдельных измерений, ни чувствительность среднего к удалению одного измерения не являются установленной минимальной ошибкой модели относительно среднего target. ΔE00 нельзя вычитать из ΔE00 для разложения ошибки.')
    d.title('1. Что восстановлено и что сохранено')
    d.paragraph('В начале этого запуска временная рабочая папка была пуста. Восстановлены исходный commit 9cad271aad159d73083259967b4107b2ce828eb1, шесть исходных файлов метаданных с точным совпадением прежних SHA256, затем только разрешённые TRAIN-фотографии. Рецепт роли TRAIN взят из неизменённого skin_mskcc_data.patient_roles и проверен по исходному split_receipt. Новые folds не создавались; private_folds.json и прежний архив C здесь отсутствуют. Для описательного анализа TRAIN fold назначения не требуются.')
    d.table(['Объект', 'Состав'], [
        ['TRAIN', '966 изображений; 24 человека; 248 уникальных patient + tag_id'],
        ['Приборные данные', '248 троек / 744 записанных assessments; 744 зависимых pairwise сравнений'],
        ['Фотографии на site', '232 site × 4; 7 × 3; 8 × 2; 1 × 1'],
        ['Тип изображения', '729 dermoscopic; 237 clinical: close-up'],
        ['Камеры', '643 iPod / 16 человек; 323 SLR / 8 человек'],
        ['Повторные фото', '1 421 пара одного exact site; 0 пар одинакового режима; 0 пар разных камер одного site']])
    d.paragraph('Файлы S1/S2 используются только для принадлежности роли, устройства и exact site. Демография не включена в отчёт. Численные endpoint-значения вне TRAIN не анализировались. Source-validation, calibration и historical test не использовались для подбора или новых оценок. Фото, прямые ISIC/patient/tag IDs, секреты и веса моделей не включены в аналитическую поставку. C и inference не переобучались и не заменялись; отсутствующие прежние checkpoint нельзя назвать заново проверенными.')
    d.title('2. Конвенции и качество reference')
    d.paragraph('Прибор: SkinColorCatch; сохранённый source_context фиксирует конвенцию производителя D65 / CIE 1964 10°. Это provenance из прежней проверки, а не индивидуальный сертификат прибора или повторная калибровка. Сравнения прибор–прибор используют CIEDE2000, kL=kC=kH=1. Наблюдаемый Lab рассчитывается из sRGB с D65 [0.95047, 1, 1.08883] / CIE 1931 2°. Прямой ΔE00 между наблюдаемым Lab 2° и приборным Lab 10° не вычислялся.')
    d.paragraph('S4 описывает 1-й, 2-й и 3-й colorimeter assessments; словарь называет таблицу анализом межоценочного согласия. В данных нет времён, операторов, переустановки прибора или независимых сессий. Поэтому измерена вариативность записанных assessments, а статистическая независимость повторов и чистый шум прибора не установлены. Все экспортированные L*, a*, b* повторов целочисленны; это дискретизация опубликованных значений, не доказательство аппаратной точности 1 Lab.')
    d.table(['|Разность координат|', 'mean','median','p90','p95','max'], [
        [c+'*', *[f(s[k]) for k in ['mean','median','p90','p95','max']]]
        for c,s in i['coordinate_absolute_pair_differences'].items()])
    d.table(['Within-site SD трёх assessments', 'mean','median','p90','p95','max'], [
        [c+'*', *[f(s[k]) for k in ['mean','median','p90','p95','max']]]
        for c,s in i['within_site_assessment_sample_sd'].items()])
    ci=i['patient_balanced']
    d.paragraph(f"Среднее pairwise ΔE00 с равным весом людей: {f(ci['patient_balanced_mean'])}; bootstrap 95% CI [{f(ci['patient_bootstrap_mean_95_ci'][0])}; {f(ci['patient_bootstrap_mean_95_ci'][1])}], 10 000 перевыборок 24 людей, seed 73019. 744 пары не трактуются как 744 независимых человека. Signed-гистограммы по каждой координате сохранены отдельно в instrument_signed_coordinate_histogram.csv.")
    d.table(['Анатомическая группа', 'sites', 'людей', 'pair mean','median','p95'], [
        [s['group'],s['sites'],s['people'],*[f(s['pairwise_delta_e00'][k]) for k in ['mean','median','p95']]]
        for s in i['strata']['anatomic_site']])
    d.title('3. Site repeatability: общая разметка вместо повторного reference')
    d.paragraph('Для каждого из 248 exact tag_id существует ровно одна строка S4. Каждая тройка S7 на фотографиях точно равна тройке этого site в S4. Между 1 421 парами фотографий одного site target ΔE00 mean=median=p95=max=0. Это буквальное повторное использование одной разметки. Независимых повторных reference-acquisitions одного site нет; временная и межсессионная site-repeatability не идентифицируется. Разные анатомические участки одного человека не объединялись.')
    d.title('4. Image repeatability: фактически измеренная вариативность capture')
    d.paragraph('JPEG: EXIF transpose, встроенный ICC → sRGB (если имеется), центральный квадрат int(0.8 × min(width,height)), исходные пиксели для среднего/медианы RGB. Отдельно Lanczos 128×128 для color36: 27 квантилей RGB, 3 mean, 3 population SD, корреляции RG/RB/GB. Это восстановленный контракт; побитовое совпадение с утраченным C-cache не заявляется. Медиана центрального crop — robust pixel statistic, а не проверенная skin-only mask. Face parser к дерматоскопии не применялся.')
    names={'original_central_mean_delta_e00':'Центральный ROI mean RGB → Lab',
           'original_central_median_delta_e00':'Центральный ROI median RGB → Lab',
           'color36_raw_rms':'color36: raw RMS (не ΔE)',
           'color36_train_z_rms':'color36: TRAIN-z RMS (не ΔE)'}
    d.table(['Разность same-site изображений', 'n pairs','mean','median','p90','p95','max'], [
        [name, capture['all_pairs'][key]['n'], *[f(capture['all_pairs'][key][k]) for k in ['mean','median','p90','p95','max']]]
        for key,name in names.items()])
    d.paragraph('Observed ΔE00 в этой таблице — различие изображение–изображение в одной конвенции 2°, не ошибка модели и не приборная цветовая ошибка. Target в каждой паре одинаков из-за общей тройки S4; это не означает, что физическая кожа и измерение повторно проверялись в момент каждого кадра.')
    d.table(['Capture pair', 'pairs','sites','людей','mean ROI ΔE mean','median','p95','median ROI ΔE median'], [
        [s['group'],s['pairs'],s['sites'],s['people'],*[f(s['original_central_mean_delta_e00'][k]) for k in ['mean','median','p95']],
         f(s['original_central_median_delta_e00']['median'])] for s in capture['strata']['mode_pair']])
    d.paragraph('NP-C: contact non-polarized dermoscopy; P-C: contact polarized dermoscopy; P-NC: non-contact polarized dermoscopy; NP-NC: clinical close-up. Это шесть сравнений разных режимов, а не тест повторяемости одинакового capture. Контакт и поляризация изменяют оптическое изображение; данные не разделяют эффекты давления, бликов, освещения, ISP и экспозиции.')
    d.table(['Группа пар', 'pairs','sites','людей','mean ROI ΔE mean','median','p95'], [
        [s['group'],s['pairs'],s['sites'],s['people'],*[f(s['original_central_mean_delta_e00'][k]) for k in ['mean','median','p95']]]
        for key in ['device','image_type_pair'] for s in capture['strata'][key]])
    d.paragraph('iPod и SLR принадлежат разным людям: различие cohort-метрик нельзя приписать только устройству. Все подробные анатомические группы, равные веса sites/людей, clipping/brightness и диагностические характеристики сохранены в capture.json. Profile/orientation counts: '+json.dumps(capture['counts']['icc_profiles'])+'; EXIF '+json.dumps(capture['counts']['exif_orientation'])+'.')
    d.table(['Site reference variability ↔ capture variability', 'n sites','Pearson r','Spearman ρ'], [
        [names.get(k,k),v['n_sites'],f(v['pearson']),f(v['spearman'])]
        for k,v in capture['instrument_vs_image_variability_site_correlations'].items() if k in names])
    d.paragraph('Это описательные корреляции 247 sites с несколькими фото, вложенных в 24 человека. Они не являются корреляциями с ошибкой C, доказательством причинности или независимыми p-values.')
    d.title('5. Oracle baselines — LEAKY_DIAGNOSTIC_ONLY')
    oracle_names={'same_site_leave_image_out':'Среднее других изображений exact site',
                  'same_site_nearest_observed':'Ближайшее observed изображение exact site',
                  'patient_other_sites':'Среднее других уникальных sites того же человека',
                  'patient_same_camera_other_sites':'Тот же человек + камера, другие sites',
                  'same_camera_anatomic_region_other_sites':'Та же камера + anatomical region, другие sites',
                  'same_site_leave_reference_observation_out':'Exact site после исключения общей reference-acquisition'}
    d.table(['Oracle', 'coverage','mean','median','p95'], [
        [oracle_names[s['name']],f(100*s['coverage'],2)+'%',*[f(s['delta_e00'][k]) for k in ['mean','median','p95']]] for s in instrument['oracles']])
    d.paragraph('Два нулевых oracle имеют один и тот же shared target: это тавтология, а не достижимая точность нового человека. Nearest выбирается по авторскому median image Lab; неизвестный observer этого поля не используется для cross-observer ΔE. Patient-oracle исключает весь target site и усредняет другие уникальные sites с равными весами, не фотографии. Camera-поправка к нему ничего не меняет из-за person–camera confounding. Эти oracle не дают формальной нижней границы deployable ошибки.')
    d.title('6. Контрфактические срезы: определения зафиксированы до чтения C errors')
    thresholds=cf['thresholds']
    d.paragraph(f"Фиксированные описательные правила: оставить sites с mean приборных pairwise ΔE ≤ site-p80 ({f(thresholds['site_reference_pair_mean_p80'])}); отдельно sites с mean pairwise median-ROI ΔE ≤ site-p80 ({f(thresholds['site_capture_pair_median_color_mean_p80'])}); отдельно изображения с high-clip + low-clip ≤ image-p80 ({f(thresholds['image_original_high_plus_low_clip_p80'],6)}). Все ties включаются. Site с одним фото не имеет измеренной capture-вариативности. Сумма high/low clipping — явно указанный proxy; пиксель с разными saturated каналами может учитываться дважды.")
    labels={'all_TRAIN':'Все TRAIN', 'lower_80pct_site_reference_variability':'Менее вариативные reference-sites',
            'lower_80pct_site_capture_variability':'Менее вариативные capture-sites',
            'lower_80pct_image_clipping_proxy':'Меньший clipping proxy',
            'reference_and_capture_stable':'Reference ∩ capture',
            'reference_capture_and_clipping_stable':'Reference ∩ capture ∩ clipping'}
    d.table(['LEAKY_DIAGNOSTIC_ONLY срез','images','sites','coverage','reference pair median','reference pair p95','C median / p95'], [
        [labels[s['name']],s['images'],s['sites'],f(100*s['coverage'],2)+'%',f(s['instrument_pairs_on_retained_sites'].get('median')),
         f(s['instrument_pairs_on_retained_sites'].get('p95')),'не вычислено: OOF отсутствует'] for s in cf['subsets']])
    d.paragraph('Эти фильтры используют reference и повторные изображения, поэтому не являются quality gate для одной фотографии. Основные 966 строк не удалялись и основная метрика не переопределялась. Снижение reference variability на отобранных sites не доказывает снижение ошибки C. Контрфактическое удаление латентного instrument noise невозможно наблюдать напрямую; не выполнялись вычитание ΔE, псевдоочистка labels или оптимистичная замена истины ближайшим к prediction reading.')
    d.title('7. Что известно об ошибке C и что пока не измерено')
    d.paragraph('Из предыдущей принятой поставки известны агрегаты C: OOF mean 5.050436, median 4.423216, p95 11.007970; >5 — 41.5114%, >10 — 7.4534%. Это исторические числа, не пересчитанные в данном аудите. Они не подменяют per-image predictions. Близкий p95 patient-oracle 10.895 не доказывает, что именно анатомия создаёт хвост C: требуется paired разбор тех же строк.')
    d.table(['Запрошенный разбор C', 'Статус'], [
        ['Patient/site/camera/type/anatomy/target L*/brightness/clipping correlations','BLOCKED: нет сохранённых OOF predictions'],
        ['Группы, создающие p95 ≈ 11; top 50 ошибок с обезличенными IDs','BLOCKED: невозможно восстановить из mean/median/p95'],
        ['C на стабильных reference/capture срезах; reference perturbation при фиксированных predictions','BLOCKED: нужен исходный per-image OOF'],
        ['Проверка C ↔ observed Lab','Будет coordinate proxy с явными разными observers; не физический cross-observer ΔE'],
        ['Основная модель и inference','Не обучались, не изменялись и не заменялись']])
    d.paragraph('Точный недостающий ресурс: LUMA_TRAIN_OOF_C.zip, файл experiment/private_review/oof_anonymized.npz. Предыдущие локальные commits 314d715 / ce64e8a и артефакты C не найдены в восстановленном GitHub. Реализован c_oof_audit.py: принимает явные названия NPZ keys, проверяет перестановку original row indices, target equality и взаимно-однозначное соответствие patient groups, затем считает требуемые residual tables и top 50. Код проверен на искусственном fixture; это проверка анализатора, не выполненный анализ C. Повторное обучение для восстановления predictions запрещено текущим заданием и не запускалось.')
    d.title('8. Решение и измеримый предел вывода')
    d.paragraph('Честный итог этого состояния: FINAL A/B/C DECISION BLOCKED, а не MODEL-LIMITED или DATA/CAPTURE-LIMITED с выдуманной долей ошибки. Приборные записи и capture действительно вариативны, но их вклад в residual C пока не установлен. Даже после получения OOF этот observational dataset не позволит без дополнительных допущений причинно разделить ошибку на точные проценты instrument, camera и estimator.')
    d.paragraph(f"Численно установлено: отдельные readings расходятся с median {f(i['pairwise_delta_e00']['median'])} и p95 {f(i['pairwise_delta_e00']['p95'])}; средний target чувствителен к удалению одного reading с median {f(i['triplet_target_sensitivity_to_omitting_one_assessment']['delta_e00']['median'])} и p95 {f(i['triplet_target_sensitivity_to_omitting_one_assessment']['delta_e00']['p95'])}; observed mean-ROI различие одного site между режимами имеет median {f(capture['all_pairs']['original_central_mean_delta_e00']['median'])} и p95 {f(capture['all_pairs']['original_central_mean_delta_e00']['p95'])}. Независимых повторных site-reference acquisitions — 0, одинаковых-mode image repeats — 0. Эти числа подтверждают ограничения измерения, но не устанавливают точность после идеального удаления шума.")
    d.paragraph('Следующее действие владельца одно: прикрепить LUMA_TRAIN_OOF_C.zip. После этого анализатор сможет закончить привязку хвоста ошибок C к sites/capture и диагностическим срезам без обучения модели, изменения inference или обращения к reserved endpoints. До paired residual анализа нет численного основания продвигать новый estimator или обещать достижение median≤2/p95≤5 на 80% фото, тем более селфи.')
    d.title('9. Воспроизведение и состав доказательств')
    d.code('git clone https://github.com/LITVA-HUB/luma-skin-vision-rnd.git\ncd luma-skin-vision-rnd\ngit checkout 9cad271aad159d73083259967b4107b2ce828eb1\n# Добавить только новые scripts/error_floor из аналитического пакета.\npython scripts/error_floor/restore_metadata.py\nPYTHONPATH=src python scripts/error_floor/instrument_audit.py --output ../error_floor\npython scripts/error_floor/restore_train_images.py\nPYTHONPATH=src python scripts/error_floor/capture_audit.py --rows ../error_floor/audit_train_rows.private.json --output ../error_floor/capture --workers 2\nPYTHONPATH=src python scripts/error_floor/reference_capture_counterfactuals.py --audit ../error_floor\npython scripts/error_floor/plot_audit.py --instrument ../error_floor/instrument.json --capture ../error_floor/capture/capture.json --output ../error_floor/figures\npython scripts/error_floor/build_report.py --audit ../error_floor --output ../error_floor/report')
    d.paragraph('CPU, Python 3.12.14; NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0, Matplotlib 3.10.8. Реальные выполненные команды и результаты запуска сохранены в COMMANDS.md и logs. MANIFEST_SHA256.json фиксирует вложенные файлы. В анонимном NPZ сохранены рассчитанные признаки и три приборных assessment, чтобы продолжить аудит без повторного скачивания фотографий. Это численные производные данных, не изображения участников и не новые instrument labels.')
    d.paragraph('Источники: исходный MSKCC skin-tone-labeling release DOI 10.34970-962049; https://isic-archive.s3.amazonaws.com/dois/10.34970-962049/mskcc-skin-tone-labeling-dataset.csv ; supplements/s1.csv, s2.csv, s4.csv, s7.csv и supplemental-data-dictionary.txt того же release. Научная публикация DOI 10.1038/s41746-025-02245-2. Первичные права, конвенции и прежние receipts: docs/data/provenance/mskcc_skin_v1 в исходном repository. Данные CC-BY; атрибуция оригинальному MSKCC/ISIC release сохраняется. Перечень прав и SHA скопирован из прежнего provenance, исходные фотографии не распространяются.')
    for path in sorted((root/'figures').glob('*.png')):
        d.figure(path)
    d.save(output)
    print(str(output/'REPORT_RU.html'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.audit,args.output)
