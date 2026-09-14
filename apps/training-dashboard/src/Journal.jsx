import React, { useState } from 'react';
import { clock, number, roles, stage, time } from './format';
import { Icon } from './icons';
export function Journal({ data, onOpen }) {
  const [role, setRole] = useState('all');
  const events = data.events.filter(e => role === 'all' || e.role === role);
  return <section className="panel journal-panel"><div className="panel-heading"><h2>Журнал обучения</h2><select aria-label="Выборка журнала" value={role} onChange={e => setRole(e.target.value)}><option value="all">Все выборки</option>{Object.entries(roles).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></div><p className="journal-note">Время фактического сохранения пакетов. Обновляется автоматически.</p><div className="events">{events.map(e => <button className="event" key={e.id} onClick={() => onOpen(e.id)}><span className="event-check"><Icon name="check" size={16}/></span><span><strong>{e.variant}<em>{e.message}</em></strong><small>{roles[e.role]} · {stage(e)} · {number(e.steps)} шагов · обучение {clock(e.seconds)}</small></span><time>{time(e.timestamp)}</time><Icon name="arrow" size={16}/></button>)}{!events.length && <div className="empty-state">Для этой выборки сохранённых пакетов ещё нет.</div>}</div></section>;
}
