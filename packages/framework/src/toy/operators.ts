const EPS = 1e-12;

export function mul(x: number, y: number): number {
    return x * y;
}

export function id(x: number): number {
    return x;
}

export function add(x: number, y: number): number {
    return x + y;
}

export function neg(x: number): number {
    return -x;
}

export function lt(x: number, y: number): number {
    return x < y ? 1 : 0;
}

export function eq(x: number, y: number): number {
    return x === y ? 1 : 0;
}

export function max(x: number, y: number): number {
    return x > y ? x : y;
}

export function isClose(x: number, y: number): number {
    return Math.abs(x - y) < 1e-2 ? 1 : 0;
}

export function sigmoid(x: number): number {
    if (x >= 0) {
        return 1 / (1 + Math.exp(-x));
    }

    const ex = Math.exp(x);
    return ex / (1 + ex);
}

export function relu(x: number): number {
    return x > 0 ? x : 0;
}

export function log(x: number): number {
    return Math.log(Math.max(x, EPS));
}

export function exp(x: number): number {
    return Math.exp(x);
}

export function logBack(x: number, d: number): number {
    return d * (1 / max(x, EPS));
}

export function inv(x: number): number {
    return 1 / x;
}

export function invBack(x: number, d: number): number {
    return -d / (x * x);
}

export function reluBack(x: number, d: number): number {
    return x > 0 ? d : 0;
}

type MapExportFn = (ls: number[]) => number[];

export function map(fn: (num: number) => number): MapExportFn {
    return (ls: number[]) => ls.map(num => fn(num));
}

export function negList(ls: number[]): number[] {
    return map(neg)(ls);
}

type ZipWithExportFn = (ls1: number[], ls2: number[]) => number[];

export function zipWith(fn: (num1: number, num2: number) => number): ZipWithExportFn {
    return (ls1: number[], ls2: number[]) => ls1.map((_, idx) => fn(ls1[idx]!, ls2[idx]!));
}

export function zip<A, B>(arr1: A[], arr2: B[]): [A, B][] {
    return arr1.map((x, i) => [x, arr2[i]!]);
}

export function addLists(ls1: number[], ls2: number[]): number[] {
    return zipWith(add)(ls1, ls2);
}

type ReduceType = (ls: number[]) => number;

export function reduce(fn: (num1: number, num2: number) => number, start: number): ReduceType {
    return (ls: number[]) => {
        let res = start;
        ls.forEach(num => {
            res = fn(num, res);
        });
        return res;
    };
}

export function sum(ls: number[]): number {
    return reduce(add, 0)(ls);
}

export function prod(ls: number[]): number {
    return reduce(mul, 1)(ls);
}

