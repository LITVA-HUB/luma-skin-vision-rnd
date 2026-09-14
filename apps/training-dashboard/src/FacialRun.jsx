import React from 'react';
import { duration, number } from './format';

const labels = { ready: 'Данные готовы', training: 'Обучается', evaluating: 'Итоговый тест', complete: 'Проверка завершена', training_complete_pending_test: 'Обучение завершено · ожидается тест', interrupted: 'Процесс остановлен', stale: 'Нет свежего прогресса', unknown: 'Статус процесса неизвестен' };

export function FacialRun({ data }) {
  if (!data) return null;
  const p = data.progress, live = data.state === 'training';
  const warmed = live && !data.warming_up;
  const finishedTraining = ['complete', 'training_complete_pending_test', 'evaluating'].includes(data.state);
  const percent = finishedTraining ? 100 : p.total_steps ? Math.min(100, p.step / p.total_steps * 100) : 0;
  const iou = data.test?.mean_image_iou ?? p.best_validation_mean_iou;
  const points = data.history.map(r => `${36 + (r.epoch - 1) / Math.max(1, (data.epochs || 12) - 1) * 680},${154 - r.iou * 124}`).join(' ');
  return <section className="facial-run" aria-label="Обучение модели выделения кожи">
    <div className="facial-heading"><div><span className="facial-kicker">НОВАЯ СЕРИЯ · ВЫДЕЛЕНИЕ КОЖИ</span><h2>{data.name}</h2><p>{number(data.train_images)} фото для обучения · {number(data.validation_images)} для подбора · {number(data.parameters == null ? null : data.parameters / 1e6, 2)} млн параметров</p></div><span className={`status ${live ? 'running' : 'completed'}`}><i/>{labels[data.state] || data.state}</span></div>
    <div className="facial-metrics"><div><span>Пройдено обучение</span><strong>{number(percent, 1)}%</strong><small>{p.epoch ? `Эпоха ${p.epoch} из ${data.epochs || p.epochs}` : 'Ожидает запуска на GPU'}</small></div><div><span>Скорость</span><strong>{warmed ? number(p.images_per_second, 1) : '—'}<em> фото/с</em></strong><small>{data.warming_up ? 'Прогрев CUDA' : 'По текущей эпохе'}</small></div><div><span>Осталось примерно</span><strong>{data.warming_up ? 'Прогрев' : warmed ? duration(p.estimated_remaining_seconds) : data.state === 'evaluating' ? 'Идёт тест' : '—'}</strong><small>Прогноз по фактическим шагам</small></div><div><span>IoU маски · {data.test ? 'итоговый тест' : 'подбор'}</span><strong>{iou == null ? '—' : `${number(iou * 100, 2)}%`}</strong><small>Совпадение с разметкой кожи</small></div></div>
    <div className="overall-progress" role="progressbar" aria-label="Прогресс обучения Seg1" aria-valuenow={percent} aria-valuemin={0} aria-valuemax={100}><span style={{ width: `${percent}%` }}/></div>
    {data.history.length > 0 && <div className="facial-chart"><span>Качество на проверочных изображениях по эпохам</span><svg viewBox="0 0 740 180" role="img" aria-label="График IoU маски кожи по эпохам"><title>IoU на выборке подбора, от 0 до 100 процентов</title>{[0, .5, 1].map(v => <g key={v}><line x1="36" x2="716" y1={154-v*124} y2={154-v*124} stroke="#e4ebe6"/><text x="0" y={158-v*124} fontSize="12" fill="#77857d">{v*100}</text></g>)}<polyline points={points} fill="none" stroke="#158160" strokeWidth="3"/>{data.history.map(r => <g key={r.epoch}><circle cx={36+(r.epoch-1)/Math.max(1,(data.epochs||12)-1)*680} cy={154-r.iou*124} r="4" fill="#158160"/><text x={36+(r.epoch-1)/Math.max(1,(data.epochs||12)-1)*680} y="177" textAnchor="middle" fontSize="11" fill="#77857d">{r.epoch}</text></g>)}</svg></div>}
    <p className="facial-note">{data.limitation}</p>
  </section>;
}
