import React from 'react';
const paths = {
  overview: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
  packages: <><path d="M3 8l3-4h12l3 4v12H3z"/><path d="M3 8h5l1 4h6l1-4h5"/></>,
  models: <><path d="M12 2l9 5v10l-9 5-9-5V7zM3 7l9 5 9-5M12 2v20M3 17l9-5 9 5"/></>,
  journal: <><path d="M14 3H5v18h14V8zM14 3v5h5M8 12h8M8 16h6"/></>,
  refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5 7a8 8 0 0 1 13-2l2 2M4 17l2 2a8 8 0 0 0 13-2"/></>,
  monitor: <><rect x="3" y="3" width="18" height="13" rx="2"/><path d="M8 21h8M12 16v5"/></>,
  arrow: <path d="M5 12h14M13 6l6 6-6 6"/>,
  down: <path d="M7 10l5 5 5-5"/>,
  download: <><path d="M12 3v12M7 10l5 5 5-5M4 15v5h16v-5"/></>,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="M16 16l5 5"/></>,
  close: <path d="M6 6l12 12M18 6L6 18"/>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v1"/></>,
  check: <path d="M5 12l4 4L19 6"/>,
};
export function Icon({ name, size = 20, className = '' }) { return <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name] || paths.info}</svg>; }
export function Logo() { return <img src="/favicon.svg" width="36" height="36" alt=""/>; }
