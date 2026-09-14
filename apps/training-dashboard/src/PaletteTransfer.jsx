import React, { useState } from 'react';
import { duration, number, roles, stage } from './format';

const STATES = {
  preparing: 'Подготовка', waiting_previous: 'Ожидает завершения HR', needs_gpu_check: 'Ожидает проверки GPU',
  ready: 'Готова к запуску', initializing: 'Запуск', training: 'Обучается', evaluating: 'Итоговый расчёт',
  complete: 'Обучение завершено · ожидается проверка', verified: 'Итоговая проверка завершена',
  failed: 'Ошибка запуска', interrupted: 'Процесс остановлен', stale: 'Нет свежего прогресса',
  unknown: 'Статус процесса неизвестен', degraded: 'Данные требуют проверки',
};
const ARMS = { original: 'Обычный старт', aligned: 'Палитра', shuffled: 'Перемешанная палитра' };
const ARCHITECTURES = { patch5m: 'Участки · 4,96 млн', soft5m: 'Уточнения · 4,85 млн', dynamic5m: 'Динамическая · 4,85 млн' };
const ISSUES = new Set(['failed', 'interrupted', 'stale', 'unknown', 'degraded']);
const ACTIVE = new Set(['initializing', 'training', 'evaluating']);

function Queue({ packages }) {
  const [role, setRole] = useState('all');
  const [variant, setVariant] = useState('all');
  const visible = packages.filter(p => (role === 'all' || p.role === role) && (variant === 'all' || p.variant === variant));
  return <details className="palette-queue">
    <summary>Состав серии и очередь <span>72 новых пакета</span></summary>
    <div className="palette-arms">
      <p><strong>Обычный старт</strong>Результаты предыдущей серии — точка сравнения.</p>
      <p><strong>Палитра</strong>Перенос основы, обученной на измеренных цветах.</p>
      <p><strong>Перемешанная палитра</strong>Контроль: помогает ли именно связь цвета с эталоном.</p>
    </div>
    <div className="palette-filters">
      <label>Выборка<select aria-label="Выборка P3" value={role} onChange={e => setRole(e.target.value)}><option value="all">Все выборки</option>{Object.entries(roles).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <label>Архитектура<select aria-label="Архитектура P3" value={variant} onChange={e => setVariant(e.target.value)}><option value="all">Все архитектуры</option>{Object.entries(ARCHITECTURES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <span className="palette-count" role="status">Показано {visible.length} из {packages.length}</span>
    </div>
    <p className="palette-scroll-hint">← Прокрутите таблицу, чтобы увидеть шаги и статус →</p>
    <div className="palette-table-scroll" tabIndex={0} role="region" aria-label="Прокручиваемая очередь P3">
      <table><caption className="palette-sr-only">Очередь новых пакетов P3; каждый пакет обучает шесть моделей</caption><thead><tr><th scope="col">Архитектура / выборка</th><th scope="col">Начальная основа</th><th scope="col">Этап</th><th scope="col">Шаги</th><th scope="col">Статус</th></tr></thead>
        <tbody>{visible.map(p => <tr key={p.id}><td><strong>{ARCHITECTURES[p.variant]}</strong><small>{roles[p.role]}</small></td><td>{ARMS[p.initialization]}</td><td>{stage(p)}</td><td className="numeric">{p.step == null ? '—' : number(p.step)} / {number(p.target_steps)}</td><td><span className={`status ${p.status}`}><i/>{p.status === 'completed' ? 'Сохранён' : p.status === 'running' ? 'Обучается' : p.current && p.step != null ? 'Последний прогресс' : 'В очереди'}</span></td></tr>)}</tbody>
      </table>
    </div>
    <p className="facial-note">54 внутренних и 18 финальных пакетов · 432 новых обучения. Для обычного старта будут использованы 36 пакетов предыдущей серии. Финальная длина обучения выбирается после внутренних проверок.</p>
  </details>;
}

export function PaletteTransfer({ data }) {
  if (!data) return null;
  const p = data.current;
  const active = ACTIVE.has(data.state), issue = ISSUES.has(data.state);
  const done = ['complete', 'verified'].includes(data.state);
  const badge = issue ? 'palette-issue' : active ? 'running' : done ? 'completed' : 'queued';
  const trainingStage = done ? 'done' : active ? 'active' : issue ? 'issue' : 'pending';
  const phases = [
    ['Подготовка на CPU', data.cpu.passed ? 'done' : 'pending', data.cpu.passed ? `${data.cpu.combinations} сочетаний проверено` : 'Проверка переноса весов'],
    ['Проверка GPU', data.gpu.passed ? 'done' : 'pending', data.gpu.passed ? 'Проверка пройдена' : data.hr_verified ? 'Можно проверять' : 'После серии HR'],
    ['Обучение на коже', trainingStage, done ? 'Пакеты сохранены' : active ? 'Идёт серия P3' : issue ? 'Нужна проверка статуса' : '72 новых пакета'],
    ['Итоговая проверка', data.native_quality_verified ? 'done' : 'pending', data.native_quality_verified ? 'Результаты проверены' : 'После обучения'],
  ];
  const eta = data.estimated_remaining_seconds != null ? duration(data.estimated_remaining_seconds) : data.state === 'waiting_previous' ? 'После HR' : done ? 'Завершено' : '—';
  const etaNote = data.estimated_remaining_seconds != null
    ? `${data.eta_basis === 'prior_native' ? 'Предварительно, по похожим пакетам AS/HR' : 'По сохранённым пакетам P3'}${data.final_steps_provisional ? ' · финальные по 2048 шагам' : ''}`
    : data.state === 'waiting_previous' ? 'После обучения и итоговой проверки HR' : active ? 'Прогноз появится после замеров' : done ? 'Все 72 результата сохранены' : issue ? 'Прогноз скрыт до подтверждения статуса' : 'Обучение ещё не началось';
  return <section className="facial-run palette-run" aria-label="Перенос палитры P3">
    <div className="facial-heading"><div><span className="facial-kicker">ПАЛИТРА → КОЖА · СЕРИЯ P3</span><h2>Luma ChromaSeed-P3</h2><p>3 архитектуры · 3 способа начать обучение · 4,85–4,96 млн параметров</p></div><span className={`status ${badge}`}><i/>{STATES[data.state] || 'Статус уточняется'}</span></div>
    <ol className="palette-phases" aria-label="Этапы серии P3">{phases.map(([label, status, note], i) => <li className={status} key={label}><span className="palette-phase-number" aria-label={status === 'done' ? 'Выполнено' : status === 'active' ? 'В процессе' : status === 'issue' ? 'Нужна проверка' : 'Ожидается'}>{status === 'done' ? '✓' : i + 1}</span><div><strong>{label}</strong><small>{note}</small></div></li>)}</ol>
    <div className="facial-metrics">
      <div><span>Сохранено пакетов</span><strong>{data.completed_banks}<em> / {data.total_banks}</em></strong><small>По фактическим результатам</small></div>
      <div><span>{issue ? 'Последние полученные шаги' : 'Шаги текущего пакета'}</span><strong>{number(p?.step)}<em>{p ? ` / ${number(p.target_steps)}` : ''}</em></strong><small>{p ? `${p.label} · ${ARMS[p.initialization]}` : 'Перенос 105 856 начальных весов'}</small></div>
      <div><span>Скорость обучения</span><strong>{number(data.steps_per_second, 1)}<em> шаг/с</em></strong><small>Один шаг обновляет 6 моделей</small></div>
      <div><span>{done ? 'Обучение' : 'Осталось примерно'}</span><strong className="palette-eta">{eta}</strong><small>{etaNote}</small></div>
    </div>
    <div className="overall-progress" role="progressbar" aria-label="Сохранённые пакеты P3" aria-valuenow={data.completed_banks} aria-valuemin={0} aria-valuemax={data.total_banks}><span style={{ width: `${data.completed_banks / data.total_banks * 100}%` }}/></div>
    {p ? <p className="facial-note">{roles[p.role]} · {stage(p)}{p.loss != null ? ` · Ошибка обучения: ${number(p.loss, 4)}` : ''}</p> : null}
    <p className="facial-note">{data.quality_claim} Прогноз времени не включает итоговую проверку.</p>
    {data.error ? <p className="palette-warning" role="alert">{data.error}</p> : null}
    {data.warnings.length ? <p className="palette-warning" role="status">{data.warnings.join(' ')}</p> : null}
    <Queue packages={data.packages}/>
  </section>;
}
