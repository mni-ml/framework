import { Scalar } from "./scalar.js";
import type { Tensor } from "../core/tensor.js";

export function centralDifference(
    f: (...args: number[]) => number,
    vals: number[],
    arg: number = 0,
    epsilon: number = 1e-6,
): number {
    const valsPlus = [...vals];
    valsPlus[arg] = valsPlus[arg]! + epsilon;

    const valsMinus = [...vals];
    valsMinus[arg] = valsMinus[arg]! - epsilon;

    return (f(...valsPlus) - f(...valsMinus)) / (2 * epsilon);
}

export class Context {
    private _savedValues: number[] = [];

    saveForBackward(...values: number[]): void {
        this._savedValues = values;
    }

    get savedValues(): number[] {
        return this._savedValues;
    }
}

export function topologicalSort(scalar: Scalar): Scalar[] {
    const visited = new Set<Scalar>();
    const sorted: Scalar[] = [];

    const dfs = (node: Scalar): void => {
        if (visited.has(node)) {
            return;
        }

        visited.add(node);
        for (const parent of node.parents) {
            dfs(parent);
        }
        sorted.push(node);
    };

    dfs(scalar);
    return sorted.reverse();
}

export function backPropagate(scalar: Scalar, dOut: number): void {
    const sorted = topologicalSort(scalar);
    const derivatives = new Map<Scalar, number>();

    derivatives.set(scalar, dOut);

    for (const node of sorted) {
        const derivative = derivatives.get(node);
        if (derivative === undefined) {
            continue;
        }

        if (node.isLeaf()) {
            node.accumulateDerivative(derivative);
            continue;
        }

        for (const [parent, grad] of node.chainRule(derivative)) {
            derivatives.set(parent, (derivatives.get(parent) ?? 0) + grad);
        }
    }
}

export function topologicalSortTensor(tensor: Tensor): Tensor[] {
    const visited = new Set<Tensor>();
    const sorted: Tensor[] = [];

    const dfs = (node: Tensor): void => {
        if (visited.has(node)) {
            return;
        }

        visited.add(node);
        for (const parent of node.parents) {
            dfs(parent);
        }
        sorted.push(node);
    };

    dfs(tensor);
    return sorted.reverse();
}

export async function backPropagateTensor(tensor: Tensor, gradOutput: Tensor): Promise<void> {
    const sorted = topologicalSortTensor(tensor);
    const gradients = new Map<Tensor, Tensor>();

    gradients.set(tensor, gradOutput);

    for (const node of sorted) {
        const grad = gradients.get(node);
        if (grad === undefined) {
            continue;
        }

        if (node.isLeaf()) {
            node.accumulateGrad(grad);
            continue;
        }

        for (const [parent, parentGrad] of await node.chainRule(grad)) {
            const existing = gradients.get(parent);
            gradients.set(parent, existing ? existing.add(parentGrad) : parentGrad);
        }
    }
}

