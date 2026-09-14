import React, { useEffect, useRef, useState } from 'react';
import { useMonitor } from './useMonitor';
import { Icon, Logo } from './icons';
import { Metrics, Current } from './Current';
import { TrainingChart } from './TrainingChart';
import { Packages } from './Packages';
import { Quality } from './Quality';
import { Journal } from './Journal';
import { FacialRun } from './FacialRun';
import { HeadRange } from './HeadRange';
import { PaletteTransfer } from './PaletteTransfer';
import { FaceTransfer } from './FaceTransfer';
import { clock, number, roles, stage, statusLabels, time } from './format';

const NAV = [['overview', 'Обзор'], ['packages', 'Пакеты'], ['models', 'Модели'], ['journal', 'Журнал']];
const TITLES = { overview: 'Обучение ChromaSeed', packages: 'Пакеты ChromaSeed', models: 'Модели ChromaSeed', journal: 'Журнал ChromaSeed' };

function PackageDialog({ item, close, onChart }) {
  const ref = useRef();
  useEffect(() => { if (item) ref.current?.showModal(); else ref.current?.close(); }, [item?.id]);
  return <dialog ref={ref} className="detail-dialog" onCancel={close} onClick={e => { if (e.target === e.currentTarget) close(); }}>
    {item && <div className="dialog-content"><div className="panel-heading"><div><h2>{item.variant}</h2><p className="muted">{item.label}</p></div><button className="icon-button" aria-label="Закрыть подробности" onClick={close}><Icon name="close"/></button></div><span className={`status ${item.status}`}><i/>{statusLabels[item.status]}</span><dl className="detail-list"><div><dt>Выборка</dt><dd>{roles[item.role]}</dd></div><div><dt>Этап</dt><dd>{stage(item)}</dd></div><div><dt>Параметры одной модели</dt><dd>{number(item.parameters)}</dd></div><div><dt>Моделей в пакете</dt><dd>6</dd></div><div><dt>Шаги</dt><dd>{item.status === 'queued' ? '—' : `${item.estimated ? '≈ ' : ''}${number(item.step)} / ${number(item.target_steps)}`}</dd></div><div><dt>Время обучения пакета</dt><dd>{item.estimated ? '≈ ' : ''}{clock(item.seconds)}</dd></div><div><dt>Сохранён</dt><dd>{time(item.completed_at)}</dd></div></dl><p className="helper">{item.estimated ? 'Число текущих шагов — оценка по времени аналогичных завершённых пакетов.' : item.steps_unknown ? 'Число шагов будет определено после внутреннего отбора настроек.' : 'Время относится ко всему пакету из шести моделей.'}</p>{item.status === 'completed' && <button className="button primary" onClick={() => onChart(item.id)}>Открыть график<Icon name="arrow" size={16}/></button>}</div>}
  </dialog>;
}

export default function App() {
  const { data, error, refreshing, refresh } = useMonitor();
  const [view, setView] = useState(NAV.some(([id]) => id === location.hash.slice(1)) ? location.hash.slice(1) : 'overview');
  const [selectedTrace, setSelectedTrace] = useState(''), [detail, setDetail] = useState(null);
  function navigate(id) { setView(id); history.replaceState(null, '', `#${id}`); window.scrollTo({ top: 0, behavior: 'instant' }); }
  const currentItem = data?.packages.find(p => p.id === detail);
  return <div className="app-shell"><aside className="sidebar"><button className="brand" onClick={() => navigate('overview')} aria-label="Luma — обзор"><Logo/><span>Luma</span></button><div className="workspace"><span>Рабочая область</span><strong>ChromaSeed</strong></div><nav aria-label="Главная навигация">{NAV.map(([id, label]) => <button key={id} aria-label={label} title={label} className={view === id ? 'active' : ''} aria-current={view === id ? 'page' : undefined} onClick={() => navigate(id)}><Icon name={id}/><span>{label}</span></button>)}</nav><div className="sidebar-bottom"><Icon name="monitor"/><div>Локальный монитор<span>localhost · только чтение</span></div></div></aside>
    <main><header className="topbar"><div className="breadcrumb">Исследования<span>/</span><strong>ChromaSeed</strong></div><div className="top-actions"><span className={`connection ${error ? 'lost' : ''}`} title={data ? `Данные получены ${time(data.sampled_at)}` : 'Подключаемся'}><i/>{error ? 'Нет связи' : data ? 'Подключено' : 'Подключение'}</span><button className="button refresh" onClick={refresh} aria-label="Обновить данные" disabled={refreshing}><Icon name="refresh" size={16} className={refreshing ? 'spinning' : ''}/><span>Обновить</span></button></div></header>
    <div className="page-heading"><h1>{TITLES[view]}</h1><p>{view === 'overview' ? 'Сравнение архитектур · до 5 млн параметров' : view === 'packages' ? 'Все внутренние и финальные запуски в одной таблице' : view === 'models' ? 'Сравнение качества и размера на сохранённых результатах' : 'Фактические события текущей серии обучения'}</p></div>
    {error && <div className="notice warning" role="alert"><Icon name="info" size={18}/>{error}</div>}
    {data ? <>{data.warnings.length > 0 && <div className="notice warning" role="status">{data.warnings.join(' ')}</div>}{['interrupted', 'unknown'].includes(data.state) && <div className="notice warning" role="status"><Icon name="info" size={18}/>Процесс обучения {data.state === 'interrupted' ? 'не запущен' : 'не удалось проверить'}. Сохранённые результаты доступны.</div>}
      {view === 'overview' && <HeadRange data={data.head_range}/>}
      {view === 'overview' && <PaletteTransfer data={data.palette_transfer}/>}
      {view === 'overview' && <FaceTransfer data={data.face_transfer}/>}
      {view === 'overview' && <FacialRun data={data.facial_skin}/>}
      {data.facial_skin && <p className="prior-series-label">Серия AS · сравнение архитектур для оценки цвета</p>}
      <Metrics data={data}/><div className="overall-progress" role="progressbar" aria-label="Фактически завершённые пакеты" aria-valuenow={data.progress.completed} aria-valuemin={0} aria-valuemax={84}><span style={{ width: `${data.progress.completed / 84 * 100}%` }}/></div>
      {view === 'overview' && <div className="overview"><div className="training-grid"><TrainingChart traces={data.traces} selected={selectedTrace} setSelected={setSelectedTrace}/><Current data={data} offline={Boolean(error)}/></div><Packages packages={data.packages} onOpen={setDetail} onAll={() => navigate('packages')}/><Quality quality={data.quality} onAll={() => navigate('models')}/></div>}
      {view === 'packages' && <Packages packages={data.packages} full onOpen={setDetail}/>}
      {view === 'models' && <Quality quality={data.quality} full/>}
      {view === 'journal' && <Journal data={data} onOpen={setDetail}/>}
      <footer className="page-footer"><span><i className={error ? 'offline' : ''}/>{error ? 'Последние данные' : 'Обновляется каждые 3 секунды'} · {time(data.sampled_at)}</span><span>Прогноз не включает независимую проверку после обучения.</span></footer>
    </> : <div className="loading-state"><div className="loading-line"/><h2>{error ? 'Монитор недоступен' : 'Подключаемся к обучению'}</h2><p>Читаем сохранённые пакеты и датчики видеокарты.</p>{error && <button className="button" onClick={refresh}>Повторить подключение</button>}</div>}
    </main><PackageDialog item={currentItem} close={() => setDetail(null)} onChart={id => { setSelectedTrace(id); setDetail(null); navigate('overview'); }}/></div>;
}
