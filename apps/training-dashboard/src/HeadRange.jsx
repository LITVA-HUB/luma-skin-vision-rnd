import React from 'react';
import { duration, number, roles } from './format';

const labels = { ready: 'Готова к запуску', training: 'Обучается', evaluating: 'Итоговый расчёт', complete: 'Обучение завершено · ожидается проверка', failed: 'Ошибка запуска', interrupted: 'Процесс остановлен', stale: 'Нет свежего прогресса', unknown: 'Статус процесса неизвестен' };
export function HeadRange({ data }) {
  if (!data) return null;
  const p = data.current, live = data.state === 'training';
  const percent = data.completed_banks / data.total_banks * 100;
  const loss = p.minibatch_loss?.length ? p.minibatch_loss.reduce((a,b)=>a+b,0)/p.minibatch_loss.length : null;
  return <section className="facial-run" aria-label="Обучение цветовой модели HR">
    <div className="facial-heading"><div><span className="facial-kicker">НОВАЯ СЕРИЯ · ТОЧНОСТЬ ЦВЕТА</span><h2>Luma ChromaSeed-HR</h2><p>7 архитектур · расширенный и линейный выход · 6 моделей в пакете</p></div><span className={`status ${live ? 'running' : 'completed'}`}><i/>{labels[data.state] || data.state}</span></div>
    <div className="facial-metrics"><div><span>Завершено пакетов</span><strong>{data.completed_banks}<em> / {data.total_banks}</em></strong><small>126 внутренних · 42 финальных</small></div><div><span>Шаги текущего пакета</span><strong>{number(p.step)}<em> / {number(p.target_steps)}</em></strong><small>{data.current_label || 'Подготовка данных'}</small></div><div><span>Скорость обучения</span><strong>{number(data.steps_per_second,1)}<em> шаг/с</em></strong><small>Один шаг обновляет 6 моделей</small></div><div><span>Осталось примерно</span><strong>{duration(data.estimated_remaining_seconds)}</strong><small>{data.final_steps_provisional ? 'Финальные пакеты пока оценены по 2048 шагам' : 'По выбранным шагам и скорости пакетов'}</small></div></div>
    <div className="overall-progress" role="progressbar" aria-label="Завершённые пакеты HR" aria-valuenow={data.completed_banks} aria-valuemin={0} aria-valuemax={data.total_banks}><span style={{width:`${percent}%`}}/></div>
    <p className="facial-note">{roles[p.role] || 'Подготовка'}{p.head_mode ? ` · ${p.head_mode === 'wide' ? 'Расширенная поправка' : 'Линейная поправка'}` : ''}{p.fold != null ? ` · Внутренняя часть ${p.fold+1}/3` : ''}{loss != null ? ` · Ошибка обучения: ${number(loss,4)}` : ''}</p>
    <p className="facial-note">{data.quality_claim} Прогноз не включает независимую проверку.</p>
    {data.error && <p role="alert" className="facial-note">{data.error}</p>}
  </section>;
}
