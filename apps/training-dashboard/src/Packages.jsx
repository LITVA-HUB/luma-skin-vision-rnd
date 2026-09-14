import React, { useMemo, useState } from 'react';
import { clock, downloadCsv, number, roles, stage, statusLabels } from './format';
import { Icon } from './icons';

export function Packages({ packages, full = false, onOpen, onAll }) {
  const [filter, setFilter] = useState('all'), [query, setQuery] = useState(''), [role, setRole] = useState('all'), [page, setPage] = useState(0);
  const rows = useMemo(() => packages.filter(p => (filter === 'all' || p.status === filter) && (role === 'all' || p.role === role) && `${p.variant} ${p.label}`.toLowerCase().includes(query.toLowerCase())).sort((a, b) => {
    const priority = { running: 0, completed: 1, queued: 2 };
    return priority[a.status] - priority[b.status] || (b.completed_at || 0) - (a.completed_at || 0);
  }), [packages, filter, query, role]);
  const size = full ? 15 : 5, maxPage = Math.max(0, Math.ceil(rows.length / size) - 1), actualPage = Math.min(page, maxPage);
  const visible = rows.slice(actualPage * size, (actualPage + 1) * size);
  return <section className="panel packages-panel">
    <div className="panel-heading wrap"><h2>Пакеты обучения</h2><div className="segmented" role="tablist" aria-label="Статус пакета">{[['all', 'Все'], ['running', 'Выполняются'], ['completed', 'Завершены'], ['queued', 'В очереди']].map(([id, label]) => <button role="tab" key={id} aria-selected={filter === id} className={filter === id ? 'selected' : ''} onClick={() => { setFilter(id); setPage(0); }}>{label}</button>)}</div>{!full && <button className="text-button" onClick={onAll}>Все {packages.length}<Icon name="arrow" size={15}/></button>}</div>
    {full && <div className="table-toolbar"><label className="search-field"><Icon name="search" size={17}/><input placeholder="Найти архитектуру" aria-label="Поиск архитектуры" value={query} onChange={e => { setQuery(e.target.value); setPage(0); }}/></label><select aria-label="Выборка пакетов" value={role} onChange={e => { setRole(e.target.value); setPage(0); }}><option value="all">Все выборки</option>{Object.entries(roles).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select><button className="button" onClick={() => downloadCsv({status:filter,role,query})}><Icon name="download" size={16}/>Скачать CSV</button></div>}
    <div className="table-scroll"><table><thead><tr><th>Архитектура</th><th>Выборка</th><th>Этап</th><th>Шаги</th><th>Время</th><th>Статус</th></tr></thead><tbody>{visible.map(p => <tr key={p.id} className={p.status === 'running' ? 'active-row' : ''} onClick={() => onOpen(p.id)}><td><button className="row-link" aria-label={`Подробнее: ${p.variant}, ${roles[p.role]}, ${stage(p)}`}>{p.variant}</button>{full && <small>{p.label}</small>}</td><td>{roles[p.role]}</td><td>{stage(p)}</td><td className="numeric">{p.status === 'queued' ? '—' : <>{p.estimated ? '≈ ' : ''}{number(p.step)}<span className="muted"> / {number(p.target_steps)}</span></>}</td><td className="numeric">{p.estimated ? '≈ ' : ''}{clock(p.seconds)}</td><td><span className={`status ${p.status}`}><i/>{statusLabels[p.status]}</span></td></tr>)}</tbody></table>{!visible.length && <div className="empty-state small">Нет пакетов с такими условиями.</div>}</div>
    <div className="table-footer"><span>{rows.length ? `${actualPage * size + 1}–${Math.min((actualPage + 1) * size, rows.length)} из ${rows.length}` : '0 пакетов'}{!full && ' · Нажмите на строку, чтобы узнать больше'}</span>{full && <div className="pagination"><button className="button" disabled={actualPage === 0} onClick={() => setPage(actualPage - 1)}>Назад</button><span>{actualPage + 1} / {maxPage + 1}</span><button className="button" disabled={actualPage === maxPage} onClick={() => setPage(actualPage + 1)}>Далее</button></div>}</div>
  </section>;
}
