// Empty by default: API and dataset images use the Vite development proxy.
// Set VITE_API_BASE_URL when serving the frontend from a separate origin.
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
