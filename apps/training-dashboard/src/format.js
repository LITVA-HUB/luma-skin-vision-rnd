export const roles = { mixed: 'Смешанная', slr_to_ipod: 'SLR → iPod', ipod_to_slr: 'iPod → SLR' };
export const number = (n, digits = 0) => n == null || !Number.isFinite(n) ? '—' : new Intl.NumberFormat('ru-RU', { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(n);
export function duration(s) {
  if (s == null) return '—';
  if (s < 60) return `${Math.ceil(s)} с`;
  const m = Math.ceil(s / 60);
  return `${m >= 60 ? `${Math.floor(m / 60)} ч` : ''}${m % 60 ? ` ${m % 60} мин` : ''}`.trim();
}
export function clock(s) {
  if (s == null) return '—';
  const t = Math.max(0, Math.floor(s));
  return `${t >= 3600 ? `${Math.floor(t / 3600)}:` : ''}${String(Math.floor(t / 60) % 60).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`;
}
export const time = n => n == null ? '—' : new Date(n * 1000).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
export const rate = n => n == null ? '—' : number(n, 5).replace(/0+$/, '');
export const stage = p => p.stage === 'inner' ? `Внутренний · ${p.fold + 1}/3` : 'Финальный';
export const statusLabels = { completed: 'Завершён', running: 'Обучается', queued: 'В очереди' };
export function downloadCsv(filters) {
  const link = document.createElement('a');
  link.href = '/api/export.csv?' + new URLSearchParams(filters);
  link.download = 'luma-training-packages.csv';
  document.body.appendChild(link);
  link.click();
  link.remove();
}
