import React, { useMemo, useState } from 'react';
import { number, roles, stage } from './format';

const COLORS = ['#168465', '#6abda2', '#4268d4', '#93a8ea', '#8650cd', '#b999e8'];
export function TrainingChart({ traces, selected, setSelected }) {
  const [mode, setMode] = useState('loss'), [hover, setHover] = useState(null);
  const trace = traces.find(t => t.id === selected) || traces[0];
  const plot = useMemo(() => {
    if (!trace) return null;
    const points = mode === 'loss' ? trace.points.map(p => ({ x: p.step, ys: p.losses })) : trace.points.slice(1).map((p, i) => ({ x: p.step, ys: [(p.step - trace.points[i].step) / (p.seconds - trace.points[i].seconds)] })).filter(p => p.ys.every(Number.isFinite));
    if (!points.length) return null;
    const maxX = Math.max(...points.map(p => p.x)), maxY = Math.max(...points.flatMap(p => p.ys)) * 1.1 || 1;
    const x = n => 55 + n / maxX * 626, y = n => 224 - n / maxY * 196;
    return { points, maxX, maxY, x, y, lines: points[0].ys.map((_, i) => points.map(p => `${x(p.x)},${y(p.ys[i])}`).join(' ')) };
  }, [trace, mode]);
  const pointed = hover == null ? null : plot?.points[hover];
  function handleMove(e) {
    if (!plot) return;
    const bounds = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - bounds.left) / bounds.width * 720 - 55) / 626 * plot.maxX;
    const index = plot.points.reduce((best, p, i) => Math.abs(p.x - x) < Math.abs(plot.points[best].x - x) ? i : best, 0);
    setHover(index);
  }
  return <section className="panel chart-panel" aria-labelledby="chart-title">
    <div className="panel-heading"><h2 id="chart-title">Ход обучения</h2><div className="line-tabs" role="tablist" aria-label="График обучения">
      {[['loss', 'Потери'], ['speed', 'Скорость']].map(([id, label]) => <button key={id} role="tab" aria-selected={mode === id} className={mode === id ? 'selected' : ''} onClick={() => { setMode(id); setHover(null); }}>{label}</button>)}
    </div></div>
    {trace ? <>
      <label className="chart-select"><span>Завершённый пакет</span><select aria-label="Пакет для графика" value={trace.id} onChange={e => { setSelected(e.target.value); setHover(null); }}>{traces.map(t => <option key={t.id} value={t.id}>{t.variant} · {roles[t.role]} · {stage(t)}</option>)}</select></label>
      <div className="chart-canvas" onMouseLeave={() => setHover(null)}>
        {plot && <svg viewBox="0 0 720 270" role="img" aria-label={mode === 'loss' ? `Фактические потери шести моделей: ${trace.variant}` : `Фактическая скорость обучения: ${trace.variant}`} onMouseMove={handleMove}>
          <title>{mode === 'loss' ? 'Потери на минипакете — фактические сохранённые значения' : 'Скорость между сохранёнными измерениями'}</title>
          {[0, 1, 2, 3, 4].map(i => <g key={i}><line x1="55" x2="681" y1={plot.y(plot.maxY * i / 4)} y2={plot.y(plot.maxY * i / 4)} stroke="#e8edea"/><text x="45" y={plot.y(plot.maxY * i / 4) + 4} textAnchor="end">{number(plot.maxY * i / 4, plot.maxY < 2 ? 2 : 1)}</text></g>)}
          {[0, 1, 2, 3, 4].map(i => <g key={i}><line x1={plot.x(plot.maxX * i / 4)} x2={plot.x(plot.maxX * i / 4)} y1="28" y2="224" stroke="#edf0ee"/><text x={plot.x(plot.maxX * i / 4)} y="246" textAnchor="middle">{number(plot.maxX * i / 4)}</text></g>)}
          {plot.lines.map((line, i) => <polyline key={i} points={line} fill="none" stroke={COLORS[i]} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round"/>)}
          <text x="367" y="267" textAnchor="middle">Шаг</text><text transform="translate(14 127) rotate(-90)" textAnchor="middle">{mode === 'loss' ? 'Потери' : 'Шагов / с'}</text>
          {pointed && <g><line x1={plot.x(pointed.x)} x2={plot.x(pointed.x)} y1="28" y2="224" stroke="#66766c" strokeDasharray="3 4"/>{pointed.ys.map((v, i) => <circle key={i} cx={plot.x(pointed.x)} cy={plot.y(v)} r="3.3" fill={COLORS[i]} stroke="white" strokeWidth="1.5"/>)}</g>}
        </svg>}
        {pointed && <div className="chart-tooltip"><strong>Шаг {number(pointed.x)}</strong>{pointed.ys.map((v, i) => <div key={i}><span><i style={{ background: COLORS[i] }}/>{mode === 'loss' ? `Запуск ${trace.slots[i]?.[0]} · ${trace.slots[i]?.[1] === .00001 ? '10⁻⁵' : '10⁻⁴'}` : 'Скорость'}</span><b>{number(v, mode === 'loss' ? 4 : 1)}</b></div>)}</div>}
      </div>
      <div className="chart-footer"><span>{mode === 'loss' ? 'Фактические потери на минипакете, не ΔE00.' : 'Один шаг обновляет шесть моделей одновременно.'}</span>{mode === 'loss' && <span className="chart-legend" title="Тёмные линии: коэффициент 10⁻⁵; светлые: 10⁻⁴. Числа обозначают начальные состояния запусков."><i style={{ background: COLORS[0] }}/> 17 <i style={{ background: COLORS[2] }}/> 29 <i style={{ background: COLORS[4] }}/> 43</span>}</div>
    </> : <div className="empty-state">График появится после сохранения первого пакета.</div>}
  </section>;
}
