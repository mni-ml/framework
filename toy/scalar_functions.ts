import { Context } from "./autodiff.js";
import * as operators from "./operators.js";

/**
 * ScalarHistory stores how a Scalar was created.
 * Note: inputs is typed as any[] to avoid circular dependency with Scalar
 */
export class ScalarHistory {
    constructor(
        public lastFn: typeof ScalarFunction | null = null,
        public ctx: Context | null = null,
        public inputs: any[] = []  // Will be Scalar[] at runtime
    ) {}
}

/**
 * Base class for all scalar operations.
 * Each operation implements forward (and later, backward).
 * 
 * Note: The apply() logic lives in Scalar class to avoid circular dependencies.
 */
export abstract class ScalarFunction {
    static forward(ctx: Context, ...inputs: number[]): number {
        throw new Error("forward not implemented");
    }

    static backward(ctx: Context, dOut: number): number[] {
        throw new Error("backward not implemented");
    }
}

export class Add extends ScalarFunction {
    static forward(ctx: Context, a: number, b: number): number {
        // Don't need to save for backward since df/da = 1 and df/db = 1
        return operators.add(a, b);
    }

    static backward(ctx: Context, dOut: number): number[] {
        return [dOut, dOut];
    }
}

export class Log extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        ctx.saveForBackward(a);
        return operators.log(a);
    }

    static backward(ctx: Context, dOut: number): number[] {
        const [a] = ctx.savedValues;
        return [dOut * (1 / Math.max(a!, 1e-12))];
    }
}

export class Mul extends ScalarFunction {
    static forward(ctx: Context, a: number, b: number): number {
        ctx.saveForBackward(a, b);
        return operators.mul(a, b);
    }

    static backward(ctx: Context, dOut: number): number[] {
        const [a, b] = ctx.savedValues;
        return [dOut * b!, dOut * a!];
    }
}

export class Inv extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        ctx.saveForBackward(a);
        return operators.inv(a);
    }

    static backward(ctx: Context, dOut: number): number[] {
        const [a] = ctx.savedValues;
        return [dOut * (-1 / a! ** 2)];
    }
}

export class Neg extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        return operators.neg(a);
    }

    static backward(ctx: Context, dOut: number): number[] {
        return [dOut * (-1)];
    }
}

export class Sigmoid extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        const result = operators.sigmoid(a);
        ctx.saveForBackward(result);
        return result;
    }

    static backward(ctx: Context, dOut: number): number[] {
        const [result] = ctx.savedValues;
        return [dOut * result! * (1 - result!)];
    }
}

export class Relu extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        ctx.saveForBackward(a);
        return operators.relu(a);
    }

    static backward(ctx: Context, dOut: number): number[] {
        const [a] = ctx.savedValues;
        return [dOut * (a! > 0 ? 1 : 0)];
    }
}

export class Exp extends ScalarFunction {
    static forward(ctx: Context, a: number): number {
        const result = operators.exp(a);
        ctx.saveForBackward(result);
        return result;
    }

    static backward(ctx: Context, dOut: number): number[] {
        const [result] = ctx.savedValues;
        return [dOut * result!];
    }
}

export class LT extends ScalarFunction {
    static forward(ctx: Context, a: number, b: number): number {
        return operators.lt(a, b);
    }

    static backward(ctx: Context, dOut: number): number[] {
        return [0, 0];
    }
}

export class EQ extends ScalarFunction {
    static forward(ctx: Context, a: number, b: number): number {
        return operators.eq(a, b);
    }

    static backward(ctx: Context, dOut: number): number[] {
        return [0, 0];
    }
}
