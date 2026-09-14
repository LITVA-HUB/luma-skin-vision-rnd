import { useCallback, useEffect, useRef, useState } from 'react';
export function useMonitor() {
  const [data, setData] = useState(null), [error, setError] = useState(null), [refreshing, setRefreshing] = useState(false);
  const controller = useRef(null);
  const refresh = useCallback(async () => {
    if (controller.current) return;
    const request = new AbortController(); controller.current = request; setRefreshing(true);
    const timeout = setTimeout(() => request.abort(), 7000);
    try {
      const response = await fetch('/api/status', { signal: request.signal, cache: 'no-store' });
      if (!response.ok) throw new Error('Сервер не отвечает');
      const next = await response.json();
      if (next.app !== 'luma-training-monitor') throw new Error('Неизвестный источник данных');
      setData(next); setError(null);
    } catch { setError('Связь с монитором потеряна. Показаны последние полученные данные.'); }
    finally { clearTimeout(timeout); controller.current = null; setRefreshing(false); }
  }, []);
  useEffect(() => { refresh(); const timer = setInterval(refresh, 3000); return () => { clearInterval(timer); controller.current?.abort(); }; }, [refresh]);
  return { data, error, refreshing, refresh };
}
