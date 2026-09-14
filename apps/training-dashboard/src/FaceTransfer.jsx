import React, { useState } from 'react';
import { duration, number } from './format';
import './FaceTransfer.css';

const ARMS = { lapa_only: 'Только LaPa', lapa_celeba: 'LaPa + CelebA' };
const STATES = {
  preparing: 'Подготовка', waiting_hr: 'В очереди: после HR и P3', waiting_p3: 'В очереди: после P3',
  needs_gpu_check: 'Ожидает проверки GPU', ready: 'Готов к запуску', initializing: 'Загрузка данных и весов',
  training: 'Обучается', finishing: 'Сохранение и завершение процесса', waiting_test: 'Ожидает тестирования',
  evaluating: 'Тестирование', waiting_audit: 'Ожидает проверки результатов', waiting_runtime: 'Ожидает замера скорости',
  benchmarking: 'Измеряется скорость на CPU', waiting_verification: 'Ожидает итогового подтверждения',
  verified: 'Результаты проверены', failed: 'Ошибка обучения', interrupted: 'Процесс остановлен',
  unknown: 'Статус процесса неизвестен', stale: 'Нет свежих данных', degraded: 'Данные требуют проверки',
};
const ISSUES = new Set(['failed', 'interrupted', 'unknown', 'stale', 'degraded']);
const AFTER_TRAIN = new Set(['waiting_test', 'evaluating', 'waiting_audit', 'waiting_runtime', 'benchmarking', 'waiting_verification', 'verified']);

function ValidationChart({ run }) {
  const points = run?.history || [];
  if (!points.length) return <div className="seg2-chart-empty"><strong>Контрольные измерения ещё не появились</strong><p>После начала обучения здесь будут два графика IoU: LaPa и CelebA.</p></div>;
  const values = points.flatMap(p => [p.lapa_iou * 100, p.celeba_iou * 100]);
  const bottom = Math.max(0, Math.floor(Math.min(...values) / 5) * 5 - 5);
  const top = Math.min(100, Math.ceil(Math.max(...values) / 5) * 5 + 5);
  const x = step => 48 + step / 5976 * 676;
  const y = value => 208 - (value * 100 - bottom) / Math.max(1, top - bottom) * 182;
  const latest = points.at(-1);
  const labels = [['lapa_iou', 'LaPa', '#198d68'], ['celeba_iou', 'CelebA', '#c77d31']];
  return <div className="seg2-chart">
    <div className="seg2-chart-caption"><strong>Совпадение маски на валидации, IoU</strong><span>На шаге {number(latest.step)} · {ARMS[run.arm]}, seed {run.seed}</span></div>
    <div className="seg2-chart-legend">{labels.map(([key, label, color]) => <span key={key}><i style={{ background: color }}/>{label}<strong>{number(latest[key] * 100, 2)}%</strong></span>)}</div>
    <svg viewBox="0 0 748 242" role="img" aria-label={`График валидации Seg2: ${ARMS[run.arm]}, seed ${run.seed}`}>
      <desc>Отображается совпадение масок кожи на двух выборках. Вертикальная шкала от {bottom} до {top} процентов IoU.</desc>
      {[bottom, (bottom + top) / 2, top].map(tick => <g key={tick}><line x1={48} x2={724} y1={y(tick / 100)} y2={y(tick / 100)} className="seg2-grid"/><text x={38} y={y(tick / 100) + 4} textAnchor="end">{number(tick, tick % 1 ? 1 : 0)}%</text></g>)}
      {[0, 1494, 2988, 5976].map(step => <g key={step}><line x1={x(step)} x2={x(step)} y1={26} y2={208} className="seg2-grid vertical"/><text x={x(step)} y={230} textAnchor={step === 5976 ? 'end' : step === 0 ? 'start' : 'middle'}>{number(step)}</text></g>)}
      {labels.map(([key, label, color]) => <g key={key}><polyline points={points.map(p => `${x(p.step)},${y(p[key])}`).join(' ')} fill="none" stroke={color} strokeWidth={2.5}/>{points.map(p => <circle key={p.step} cx={x(p.step)} cy={y(p[key])} r={4} fill={color} tabIndex={0} aria-label={`${label}, шаг ${p.step}: ${number(p[key] * 100, 2)} процентов IoU`}><title>{label} · шаг {p.step}: {number(p[key] * 100, 2)}% IoU</title></circle>)}</g>)}
    </svg>
    <p className="facial-note">По горизонтали — обновления весов. Точки сравнения: 1 494, 2 988 и 5 976. Валидация помогает выбрать модель; итоговый тест проводится отдельно.</p>
  </div>;
}

function Details({ data }) {
  const [arm, setArm] = useState('all');
  const [seed, setSeed] = useState('all');
  const [chart, setChart] = useState('');
  const visible = data.runs.filter(r => (arm === 'all' || r.arm === arm) && (seed === 'all' || String(r.seed) === seed));
  const selected = data.runs.find(r => r.id === chart) || data.runs.find(r => r.id === data.current?.id) || data.runs[0];
  return <details className="palette-queue seg2-details">
    <summary>Данные, очередь и графики <span>6 дообучений</span></summary>
    <div className="seg2-data-grid">{[['lapa', 'LaPa', 'Исходная обучающая выборка'], ['celeba', 'CelebAMask-HQ', 'Дополнительные фото после очистки']].map(([id, label, note]) => <article key={id}><span>{note}</span><h3>{label}</h3><strong>{number(data.counts[id]?.train)}<small>обучающих фото</small></strong><p>Валидация: {number(data.counts[id]?.validation)} · Тест: {number(data.counts[id]?.test)}</p></article>)}</div>
    <p className="facial-note">Каждый батч: 16 общих примеров LaPa и 16 дополнительных — из LaPa или CelebA. Размер батча и число шагов одинаковы. Seed задаёт порядок примеров и случайных преобразований.</p>
    <div className="palette-filters seg2-filters">
      <label>Данные<select aria-label="Данные Seg2" value={arm} onChange={e => setArm(e.target.value)}><option value="all">Оба состава</option>{Object.entries(ARMS).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
      <label>Seed<select aria-label="Seed Seg2" value={seed} onChange={e => setSeed(e.target.value)}><option value="all">Все три</option>{[17, 29, 43].map(value => <option value={value} key={value}>{value}</option>)}</select></label>
      <span className="palette-count" role="status">Показано {visible.length} из {data.runs.length}</span>
    </div>
    <p className="palette-scroll-hint">← Прокрутите таблицу, чтобы увидеть все столбцы →</p>
    <div className="palette-table-scroll seg2-table-scroll" tabIndex={0} role="region" aria-label="Прокручиваемая очередь Seg2"><table><caption className="palette-sr-only">Шесть дообучений Seg2: два состава данных и три seed</caption><thead><tr><th scope="col">Данные / Seed</th><th scope="col">Обновления весов</th><th scope="col">Последняя валидация</th><th scope="col">Статус</th><th scope="col">График</th></tr></thead><tbody>{visible.map(run => {
      const last = run.history.at(-1);
      return <tr key={run.id}><td><strong>{ARMS[run.arm]}</strong><small>Seed {run.seed}</small></td><td className="numeric">{number(run.step)} / {number(run.target_steps)}</td><td className="numeric">{last ? `${number(last.macro_iou * 100, 2)}% IoU` : 'Ещё нет'}{last ? <small>Среднее двух источников</small> : null}</td><td><span className={`status ${run.status}`}><i/>{run.status === 'completed' ? 'Сохранён' : run.status === 'running' ? 'Обучается' : run.step != null ? 'Последние данные' : 'В очереди'}</span></td><td><button className="button seg2-chart-button" onClick={() => setChart(run.id)} aria-label={`График ${ARMS[run.arm]}, seed ${run.seed}`}>{selected?.id === run.id ? 'Выбран' : 'Открыть'}</button></td></tr>;
    })}</tbody></table></div>
    <div className="seg2-chart-select"><label>График обучения<select aria-label="График Seg2" value={selected?.id || ''} onChange={e => setChart(e.target.value)}>{data.runs.map(run => <option key={run.id} value={run.id}>{ARMS[run.arm]} · seed {run.seed}</option>)}</select></label></div>
    <ValidationChart run={selected}/>
    <p className="facial-note">LaPa TEST уже использовался раньше. Отсутствие одинаковых людей между двумя источниками не подтверждено. Исходная Seg1 участвует в выборе: дополнительное обучение может и ухудшить результат.</p>
  </details>;
}

export function FaceTransfer({ data }) {
  if (!data) return null;
  const issue = ISSUES.has(data.state);
  const training = data.state === 'training';
  const done = AFTER_TRAIN.has(data.state);
  const progress = Math.min(data.total_updates, Math.max(0, data.observed_updates));
  const phases = [
    ['Подготовка', data.cpu.passed ? 'done' : 'pending', data.cpu.passed ? '6 CPU-проверок пройдено' : 'Данные и исходные веса'],
    ['Проверка GPU', data.gpu.passed ? 'done' : 'pending', data.gpu.passed ? 'Проверка пройдена' : 'После HR и P3'],
    ['Дообучение', done ? 'done' : training ? 'active' : issue ? 'issue' : 'pending', `${data.completed_runs} из 6 запусков сохранено`],
    ['Итоговая проверка', data.verified ? 'done' : ['evaluating', 'benchmarking'].includes(data.state) ? 'active' : 'pending', data.verified ? 'Результаты подтверждены' : 'Тест, аудит и замер скорости'],
  ];
  const eta = data.estimated_remaining_seconds != null ? duration(data.estimated_remaining_seconds) : data.state === 'waiting_hr' ? 'После HR и P3' : data.state === 'waiting_p3' ? 'После P3' : done ? 'Завершено' : '—';
  return <section className="facial-run palette-run seg2-run" aria-label="Дообучение Seg2">
    <div className="facial-heading"><div><span className="facial-kicker">БОЛЬШЕ ДАННЫХ · СЕРИЯ SEG2</span><h2>Luma ChromaSeed-Seg2</h2><p>{number(data.parameters == null ? null : data.parameters / 1e6, 2)} млн параметров · {number(data.training_images)} доступных обучающих фото</p></div><span className={`status ${issue ? 'palette-issue' : training ? 'running' : done ? 'completed' : 'queued'}`}><i/>{STATES[data.state] || 'Статус уточняется'}</span></div>
    <ol className="palette-phases" aria-label="Этапы серии Seg2">{phases.map(([label, status, note], index) => <li className={status} key={label}><span className="palette-phase-number" aria-label={status === 'done' ? 'Выполнено' : status === 'active' ? 'В процессе' : status === 'issue' ? 'Нужна проверка' : 'Ожидается'}>{status === 'done' ? '✓' : index + 1}</span><div><strong>{label}</strong><small>{note}</small></div></li>)}</ol>
    <div className="facial-metrics seg2-metrics">
      <div><span>Сохранено обучений</span><strong>{data.completed_runs}<em> / {data.total_runs}</em></strong><small>Два состава данных × три seed</small></div>
      <div><span>{issue ? 'Подтверждённые шаги' : 'Обновления весов'}</span><strong>{number(progress)}<em> / {number(data.total_updates)}</em></strong><small>{data.current ? `${ARMS[data.current.arm]} · seed ${data.current.seed}` : 'Три бюджета сравниваются в каждой траектории'}</small></div>
      <div><span>Скорость обучения</span><strong>{number(data.images_per_second, 1)}<em> фото/с</em></strong><small>С учётом загрузки и валидации</small></div>
      <div><span>{done ? 'Дообучение' : 'Осталось примерно'}</span><strong className="seg2-eta">{eta}</strong><small>{data.estimated_remaining_seconds != null ? 'По текущему запуску · без итоговой проверки' : issue ? 'Прогноз скрыт до подтверждения статуса' : done ? 'Следующий этап показан выше' : 'Прогноз появится по реальному обучению'}</small></div>
    </div>
    <div className="overall-progress" role="progressbar" aria-label="Шаги серии Seg2" aria-valuemin={0} aria-valuemax={data.total_updates} aria-valuenow={progress}><span style={{ width: `${progress / data.total_updates * 100}%` }}/></div>
    {data.current?.loss != null ? <p className="facial-note">Ошибка текущего обучающего батча: {number(data.current.loss, 4)}. Она не равна итоговому IoU или ошибке цвета.</p> : null}
    {data.quality ? <div className="seg2-verified-quality"><strong>Выбранная модель · проверенный тест</strong><span>LaPa: {number(data.quality.iou.lapa * 100, 2)}% IoU</span><span>CelebA: {number(data.quality.iou.celeba * 100, 2)}% IoU</span></div> : null}
    <p className="facial-note">{data.limitation}</p>
    {data.error ? <p className="palette-warning" role="alert">{data.error}</p> : null}
    {data.warnings.length ? <p className="palette-warning" role="status">{data.warnings.join(' ')}</p> : null}
    <Details data={data}/>
  </section>;
}
