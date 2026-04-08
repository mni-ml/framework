import { Tensor } from "./tensor.js";

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
        for (const module of this.children()) {
            module.eval();
        }
    }
}

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
        if (this.value instanceof Tensor) {
            return this.value.grad;
        }
        return null;
    }

    update(v: T) {
        this.value = v;
    }
}
