/* 
Modules form a tree that store parameters and other
submodules. They make up the basis of neural network stacks.

Attributes:
    modules : Storage of the child modules
    parameters : Storage of the module's parameters
    training : Whether the module is in training mode or evaluation mode
*/

import type { Tensor } from "./tensor.js";
import { Scalar } from "./scalar.js";

export class Module<P extends BaseParameter = BaseParameter> {
    protected _modules: Record<string, Module<P>> = {};
    protected _parameters: Record<string, P> = {};
    training: boolean = true;

    constructor() {
        return new Proxy(this, {
            set: (target, key: string | symbol, value, receiver) => {
                if (value instanceof Module) {
                    target._modules[key as string] = value;
                } 
                else if (value instanceof BaseParameter) {
                    target._parameters[key as string] = value as P;
                }

                return Reflect.set(target, key, value, receiver);
            }
        })
    }

    parameters(): P[] {
        let params: P[] = [];

        for (const p of Object.values(this._parameters)) {
            params.push(p);
        }

        for (const m of Object.values(this._modules) as Module<P>[]) {
            params.push(...m.parameters());
        }

        return params;
    }

    namedParameters(): Array<[string, P]> {
        const named: Array<[string, P]> = Object.entries(this._parameters);
    
        for (const [moduleName, module] of Object.entries(this._modules)) {
            for (const [name, param] of module.namedParameters()) {
                named.push([`${moduleName}.${name}`, param]);
            }
        }
    
        return named;
    }

    modules(): Module<P>[] {
        const all: Module<P>[] = [this];
        for (const child of this.children()) {
            all.push(...child.modules());
        }
        return all;
    }

    children(): Module<P>[] {
        return Object.values(this._modules);
    }

    train(): void {
        this.training = true;
        for (const module of this.children()) {
            module.train();
        }
    }

    eval(): void {
        this.training = false;
        for (const module of this.modules()) {
            module.eval();
        }
    }
}

// Non-generic base class to type Parameter class yet not Module class
export abstract class BaseParameter {
    name?: string | undefined;
}

export class Parameter<T=Tensor> extends BaseParameter {
    value: T;

    constructor(value: T, name?: string) {
        super();
        this.value = value;

        if (name) {
            this.name = name;
        }
    }

    get grad() {
        if (this.value instanceof Scalar) {
            return this.value.derivative ?? 0;
        }
        return 0;
    }

    update(v: T) {
        this.value = v;
    }
}
