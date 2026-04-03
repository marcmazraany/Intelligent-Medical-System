import { useEffect, useState } from 'react';

export function useLocalStorageState<T>(
  key: string,
  initialValue: T,
  opts?: { serialize?: (v: T) => string; deserialize?: (raw: string) => T }
) {
  const serialize = opts?.serialize ?? ((v: T) => JSON.stringify(v));
  const deserialize = opts?.deserialize ?? ((raw: string) => JSON.parse(raw) as T);

  const [state, setState] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw ? deserialize(raw) : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, serialize(state));
    } catch {
      // ignore (storage full / blocked)
    }
  }, [key, serialize, state]);

  return [state, setState] as const;
}
