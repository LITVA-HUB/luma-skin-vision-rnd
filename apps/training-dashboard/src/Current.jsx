import React from 'react';
import { clock, duration, number, roles, time } from './format';

export function Metrics({ data }) {
  const speed = data.current?.steps_per_second;
  return <div className="metrics-strip">
    <div className="metric"><span>Завершено пакетов</span><strong>{data.progress.completed}<em> / {data.progress.total}</em></strong><small>{data.progress.inner_completed} / 63 внутренних · {data.progress.final_completed} / 21 финальный</small></div>
    <div className="metric"><span>Осталось примерно</span><strong>{data.eta.seconds === 0 ? 'Готово' : duration(data.eta.seconds)}</strong><small title="Оценочный диапазон, не статистический доверительный интервал">{data.eta.seconds ? `${duration(data.eta.low_seconds)} – ${duration(data.eta.high_seconds)}` : 'До завершения обучения'}</small></div>
    <div className="metric"><span>Скорость обучения</span><strong>{number(speed, 1)}<em> шага/с</em></strong><small>По завершённым пакетам этой архитектуры</small></div>
    <div className="metric"><span>Загрузка GPU</span><strong>{number(data.gpu.utilization)}<em>{data.gpu.utilization != null ? '%' : ''}</em></strong><small>{data.gpu.name || 'Нет данных датчиков'}</small></div>
  </div>;
}

export function Current({ data, offline }) {
  const current = data.current, gpu = data.gpu;
  const progress = current ? current.step / current.target_steps * 100 : 0;
  const memory = gpu.memory_total_mb ? gpu.memory_used_mb / gpu.memory_total_mb * 100 : 0;
  const title = current ? 'Сейчас обучается' : ['complete', 'verified'].includes(data.state) ? 'Обучение завершено' : 'Состояние обучения';
  const messages = { selection: 'Выбираются настройки по внутренним результатам.', evaluation: 'Обучение пакетов закончено. Рассчитываются итоговые метрики.', complete: 'Все пакеты сохранены. Независимая проверка — следующий этап.', verified: 'Результаты обучения проверены.', interrupted: 'Процесс обучения сейчас не запущен. Сохранённые результаты доступны.', unknown: 'Не удалось определить состояние процесса.', degraded: 'Часть данных пока недоступна. Ожидаем обновления.', not_started: 'Журнал запуска ещё не появился.' };
  return <section className="panel current-panel"><div className="panel-heading"><h2>{title}</h2>{current && <span className={`live-mark ${offline ? 'offline' : ''}`} title={offline ? 'Последние данные' : 'Процесс работает'}/>}</div>
    {current ? <><dl className="current-details"><div><dt>Семейство</dt><dd>{current.variant}</dd></div><div><dt>Роль</dt><dd>{roles[current.role]}</dd></div><div><dt>{current.stage === 'inner' ? 'Разбиение' : 'Этап'}</dt><dd>{current.stage === 'inner' ? `${current.fold + 1} из 3` : 'Финальное обучение'}</dd></div></dl><div className="progress-line"><div className="progress-track" role="progressbar" aria-label="Оценочный прогресс текущего пакета" aria-valuenow={current.step} aria-valuemin={0} aria-valuemax={current.target_steps}><span style={{ width: `${progress}%` }}/></div><b>≈ {number(current.step)} <span>/ {number(current.target_steps)}</span></b></div><p className="helper">{current.overdue ? 'Ожидаем подтверждения завершения пакета.' : 'Прогресс текущего пакета оценочный.'}</p></> : <p className="state-message">{messages[data.phase] || messages[data.state]}</p>}
    <div className="gpu-details"><div className="label-line"><span>Память GPU</span></div><div className="progress-line"><div className="progress-track memory-track"><span style={{ width: `${memory}%` }}/></div><b>{number(gpu.memory_used_mb == null ? null : gpu.memory_used_mb / 1024, 1)} <span>/ {number(gpu.memory_total_mb == null ? null : gpu.memory_total_mb / 1024, 1)} ГБ</span></b></div><dl><div><dt>Температура</dt><dd>{number(gpu.temperature)} °C</dd></div><div><dt>Мощность</dt><dd>{number(gpu.power_w)} <span className="muted">/ {number(gpu.power_limit_w)} Вт</span></dd></div></dl></div>
    <div className="current-footer"><span>Прошло {clock(data.elapsed_seconds)}</span><span title="Прогноз может меняться">{data.eta.finish_at ? `Финиш ≈ ${time(data.eta.finish_at).slice(0, 5)}` : 'Время местное'}</span></div>
  </section>;
}
