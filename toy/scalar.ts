import { backPropagate, Context } from "./autodiff.js";
import { zip } from "./operators.js"
import {
    ScalarHistory, 
    ScalarFunction,
    Add,
    Log,
    Mul,
    Inv,
    Neg,
    Sigmoid,
    Relu,
    Exp,
    LT,
    EQ,
} from "./scalar_functions.js";

export type ScalarLike = number | Scalar;
export type GradPair = [Scalar, number];

let _varCount = 0;

/**
 * Scalar: A number that tracks its computation history.
 * Behaves like a regular number but records operations for autodiff.
 */
export class Scalar {
    data: number;
    readonly history: ScalarHistory | null;
    readonly uniqueId: number;
    readonly name: string;

    derivative: number | null = null; // Filled in during backward pass

    constructor(
        value: number,  
        history: ScalarHistory | null = null,
        name?: string
    ) {
        _varCount++;
        this.uniqueId = _varCount;
        this.data = value;
        this.history = history;
        this.name = name ?? `var${this.uniqueId}`;
    }

    toString(): string {
        return `Scalar(${this.data})`;
    }

    /**
     * Apply a ScalarFunction to the given values.
     * Handles unwrapping Scalars to numbers, calling forward, and wrapping the result.
     */
    static apply(fn: typeof ScalarFunction, ...vals: ScalarLike[]): Scalar {
        const rawVals: number[] = [];
        const scalars: Scalar[] = [];

        for (const v of vals) {
            if (v instanceof Scalar) {
                scalars.push(v);
                rawVals.push(v.data);
            } else {
                scalars.push(new Scalar(v));
                rawVals.push(v);
            }
        }

        const ctx = new Context();

        const result = fn.forward(ctx, ...rawVals);

        const history = new ScalarHistory(fn, ctx, scalars);
        return new Scalar(result, history);
    }

    add(b: ScalarLike): Scalar {
        return Scalar.apply(Add, this, b);
    }

    mul(b: ScalarLike): Scalar {
        return Scalar.apply(Mul, this, b);
    }

    div(b: ScalarLike): Scalar {
        return Scalar.apply(Mul, this, Scalar.apply(Inv, b));
    }

    rdiv(b: ScalarLike): Scalar {
        return Scalar.apply(Mul, b, Scalar.apply(Inv, this));
    }

    sub(b: ScalarLike): Scalar {
        return Scalar.apply(Add, this, Scalar.apply(Neg, b));
    }

    neg(): Scalar {
        return Scalar.apply(Neg, this);
    }

    lt(b: ScalarLike): Scalar {
        return Scalar.apply(LT, this, b);
    }

    eq(b: ScalarLike): Scalar {
        return Scalar.apply(EQ, this, b);
    }

    gt(b: ScalarLike): Scalar {
        return Scalar.apply(LT, b, this);
    }

    log(): Scalar {
        return Scalar.apply(Log, this);
    }

    exp(): Scalar {
        return Scalar.apply(Exp, this);
    }

    sigmoid(): Scalar {
        return Scalar.apply(Sigmoid, this);
    }

    relu(): Scalar {
        return Scalar.apply(Relu, this);
    }

    chainRule(dOut: number): Iterable<GradPair> {
        const h = this.history;
        if (!h) throw new Error("Missing scalar history");
        if (!h.lastFn) throw new Error("Missing lastFn in scalar history");
        if (!h.ctx) throw new Error("Missing ctx in scalar history");
        if (!h.inputs) throw new Error("Missing inputs in scalar history");

        const gradients: number[] = h.lastFn.backward(h.ctx, dOut);

        const inputs = h.inputs as Scalar[];
        return zip(inputs, gradients);
    }

    backward(dOut?: number) {
        if (dOut == null || dOut == undefined) {
            dOut = 1.0;
        }
        backPropagate(this, dOut);
    }

    isLeaf(): boolean {
        return !this.history?.lastFn;
    }

    isConstant(): boolean {
        return !this.history;
    }

    get parents(): Scalar[] {
        return this.history?.inputs ?? [];
    }

    accumulateDerivative(d: number): void {
        if (!this.isLeaf()) {
            throw new Error("Cannot accumulate derivative of a non-leaf scalar");
        }
        if (this.derivative === null) {
            this.derivative = 0;
        }
        this.derivative += d;
    }
}

export { ScalarHistory };
