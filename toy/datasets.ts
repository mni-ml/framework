export type Point = [number, number];

export interface Graph {
  N: number; // size of data
  X: Point[];
  y: number[];
}

/** generate N random 2D points in [0, 1) × [0, 1) */
export function makePts(N: number): Point[] {
  const X: Point[] = [];
  for (let i = 0; i < N; i++) {
    const x1 = Math.random();
    const x2 = Math.random();
    X.push([x1, x2]);
  }
  return X;
}

export function simple(N: number): Graph {
  const X = makePts(N);
  const y = X.map(([x1]) => (x1 < 0.5 ? 1 : 0));
  return { N, X, y };
}

export function diag(N: number): Graph {
  const X = makePts(N);
  const y = X.map(([x1, x2]) => (x1 + x2 < 0.5 ? 1 : 0));
  return { N, X, y };
}

export function split(N: number): Graph {
  const X = makePts(N);
  const y = X.map(([x1]) => (x1 < 0.2 || x1 > 0.8 ? 1 : 0));
  return { N, X, y };
}

export function xor(N: number): Graph {
  const X = makePts(N);
  const y = X.map(([x1, x2]) =>
    (x1 < 0.5 && x2 > 0.5) || (x1 > 0.5 && x2 < 0.5) ? 1 : 0
  );
  return { N, X, y };
}

export function circle(N: number): Graph {
  const X = makePts(N);
  const y = X.map(([x1, x2]) => {
    const dx = x1 - 0.5;
    const dy = x2 - 0.5;
    return dx * dx + dy * dy > 0.1 ? 1 : 0;
  });
  return { N, X, y };
}

export function spiral(N: number): Graph {
  const half = Math.floor(N / 2);

  const fx = (t: number) => (t * Math.cos(t)) / 20.0;
  const fy = (t: number) => (t * Math.sin(t)) / 20.0;

  const X1: Point[] = Array.from({ length: half }, (_, i) => {
    const t = 10.0 * (i / half);
    return [fx(t) + 0.5, fy(t) + 0.5];
  });

  const X2: Point[] = Array.from({ length: half }, (_, i) => {
    const t = -10.0 * (i / half);
    return [fy(t) + 0.5, fx(t) + 0.5];
  });

  const X = [...X1, ...X2];
  const y = [...Array(half).fill(0), ...Array(half).fill(1)];

  return { N, X, y };
}

export const datasets: Record<string, (N: number) => Graph> = {
  Simple: simple,
  Diag: diag,
  Split: split,
  Xor: xor,
  Circle: circle,
  Spiral: spiral,
};  
