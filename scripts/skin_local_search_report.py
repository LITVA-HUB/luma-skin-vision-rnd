"""Aggregate local frozen experiment receipts without participant identifiers."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/skin_local_search_v1"
OUT = ROOT / "docs/benchmarks/skin_local_search_v1"


def main():
    evaluation = json.loads((RUN / "evaluation.json").read_text(encoding="utf-8"))
    inner = json.loads((RUN / "inner_records.json").read_text(encoding="utf-8"))
    rows = []
    for protocol in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for method in ("ridge", "krr", "random_rbf", "guided_rbf", "mlp"):
            records = [r for r in evaluation["models"] if r["protocol"] == protocol and r["method"] == method]
            row = {"protocol": protocol, "method": method, "n_seeds": len(records)}
            for key in ("person_mean", "image_mean", "site_person_mean", "median", "p90", "gt5", "gt10"):
                row[key] = float(np.mean([r["outer_metrics"][key] for r in records]))
            for key in ("fit_seconds", "numeric_bytes", "numeric_scalars", "serialized_bytes", "peak_allocated_bytes",
                        "batch1_cpu_us_p50", "batch1_cpu_us_p95", "inner_person_mean"):
                row[key] = float(np.mean([r[key] for r in records]))
            row["train_person_mean"] = float(np.mean([r["train_metrics"]["person_mean"] for r in records]))
            row["outer_seed_person_means"] = [r["outer_metrics"]["person_mean"] for r in records]
            row["selected_parameter"] = records[0]["parameter"]
            rows.append(row)
    training = {m: {"inner_fits": len([r for r in inner if r["method"] == m]),
                    "inner_fit_seconds_sum": sum(r["fit_seconds"] for r in inner if r["method"] == m)}
                for m in ("ridge", "krr", "random_rbf", "guided_rbf", "mlp")}
    result = {"evidence": evaluation["evidence"], "primary": "person-balanced native Lab DeltaE00; lower is better",
              "rows": rows, "training": training,
              "total_fit_phase_seconds": json.loads((RUN / "fit_completion.json").read_text())["seconds_this_invocation"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "source_lock.json").write_bytes((RUN / "source_lock.json").read_bytes())
    lines = ["# Компактное обучение с поиском по ошибке: измерения", "",
             "13 сентября 2026. RTX 4060 8 GB. Только исходный TRAIN: 966 изображений / 24 человека.", "",
             "Это исследовательская проверка на ранее использовавшихся данных. Три протокола пересекаются; результаты не являются тремя независимыми подтверждениями. Старые validation/calibration/test не открывались.", "",
             "Все значения — среднее метрик отдельных моделей по seeds, без ансамбля. Основная ошибка — CIEDE2000, сначала средняя внутри человека, затем между людьми. Меньше лучше.", "",
             "| Протокол | Метод | Внутренний ΔE00 | Отложенные люди ΔE00 | Среднее по фото | Числовые байты | Обучение, с |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['protocol']} | {r['method']} | {r['inner_person_mean']:.4f} | {r['person_mean']:.4f} | {r['image_mean']:.4f} | {r['numeric_bytes']:.0f} | {r['fit_seconds']:.4f} |")
    lines += ["", "## Что именно измерено", "",
              "Mixed: 18 человек / 734 фото для обучения, 6 / 232 для внешней исследовательской проверки. SLR→iPod: 8 / 323 → 16 / 643; iPod→SLR — обратное направление. Настройки каждого метода выбирались по трём внутренним фолдам с разделением по людям; все настройки и итоговые веса зафиксированы до внешней проверки.", "",
              "RBF и MLP хранят ровно 2749 чисел / 10 996 байт FP32, включая нормализацию. RBF хранит центры из обучающих признаков, ширины и коэффициенты; они включены в размер. Файлы NPZ больше полезной числовой нагрузки из-за служебных заголовков.", "",
              "Guided RBF перебирает до 768 кандидатов на каждом из 64 шагов через Schur complement и пересчитывает выходные веса аналитически, без обратного распространения ошибки. Это до 49 152 оценок условного улучшения, без отдельного полного обучения для каждого кандидата. Random RBF использует тот же банк и размер. MLP — 36→64 SiLU→3 плюс линейный путь, AdamW, 512 шагов.", "",
              "Тайминг включает нормализацию, построение банка, поиск и вычисление весов; CUDA синхронизирована. Вход — уже рассчитанные 36 статистик цвета. Извлечение лица/области кожи и признаков, чтение данных, подбор всех конфигураций и упаковка модели не входят во время одного fit. Полное время фазы с подбором указано отдельно.", "",
              f"Всего 297 внутренних fits и 33 итоговых fits; вся фаза заняла {result['total_fit_phase_seconds']:.2f} с. Отдельный первый синтетический прогрев не входит в этот запуск. Порядок методов фиксирован, поэтому холодный старт может затронуть первые fits. Сравнение времени итоговых fits использует уже прогретый процесс.", "",
              "| Протокол | Метод | Файл NPZ, байт | CPU batch1 p50, мкс | CPU batch1 p95, мкс | Peak CUDA allocated, MiB |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['protocol']} | {r['method']} | {r['serialized_bytes']:.0f} | {r['batch1_cpu_us_p50']:.2f} | {r['batch1_cpu_us_p95']:.2f} | {r['peak_allocated_bytes']/2**20:.2f} |")
    lines += ["", "CPU latency — только функция модели по готовым признакам, NumPy FP32, один поток, 30 прогревов и 200 замеров. Это не камера→результат и не мобильный benchmark. GPU peak — память PyTorch, не весь расход рабочего стола/драйвера.", "",
              "## Проверяемость и пределы", "",
              "Независимые NumPy/exhaustive тесты подтверждают решение ridge, величину каждого greedy-улучшения, устойчивость на дублированных атомах, CPU/GPU согласованность. Отдельные проверки подтверждают роли, веса человек→участок→снимок, неизменность выбора при подмене внешних меток и сериализацию FP32.", "",
              "Семейство методов имеет прямые предшественники: [OMP](https://ieeexplore.ieee.org/document/4385788), [Stochastic Configuration Networks](https://ieeexplore.ieee.org/document/8013920/), [ELM](https://doi.org/10.1016/j.neucom.2005.12.126). Это не заявление об изобретении общего алгоритма. Не проверялись идентификация человека и точность на обычных селфи; здесь измеряется цвет кожи по подготовленным областям изображения и инструментальному Lab.", "",
              "[Протокол](../../research/skin_local_search_v1_protocol.md) · [Точные агрегаты](summary.json) · [Хеши источников](source_lock.json). Локальные веса и построчные результаты лежат в gitignored experiments/runs/skin_local_search_v1; изображения и идентификаторы в отчёт не включены.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
