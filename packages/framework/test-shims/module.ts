type ScalarLike = {
  derivative?: number | null;
};

type TensorLike = {
  grad?: unknown;
};

export class Module<P extends BaseParameter = BaseParameter> {
  protected _modules: Record<string, Module<P>> = {};
  protected _parameters: Record<string, P> = {};
  training = true;

  constructor() {
    return new Proxy(this, {
      set: (target, key: string | symbol, value, receiver) => {
        if (value instanceof Module) {
          target._modules[key as string] = value;
        } else if (value instanceof BaseParameter) {
          target._parameters[key as string] = value as P;
        }

        return Reflect.set(target, key, value, receiver);
      },
    });
  }

  parameters(): P[] {
    const params: P[] = [...Object.values(this._parameters)];

    for (const module of Object.values(this._modules) as Module<P>[]) {
      params.push(...module.parameters());
    }

    return params;
  }

  namedParameters(): Array<[string, P]> {
    const named: Array<[string, P]> = Object.entries(this._parameters);

    for (const [moduleName, module] of Object.entries(this._modules)) {
      for (const [name, parameter] of module.namedParameters()) {
        named.push([`${moduleName}.${name}`, parameter]);
      }
    }

    return named;
  }

  modules(): Module<P>[] {
    const modules: Module<P>[] = [this];
    for (const child of this.children()) {
      modules.push(...child.modules());
    }
    return modules;
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
  name?: string;
}

export class Parameter<T = unknown> extends BaseParameter {
  value: T;

  constructor(value: T, name?: string) {
    super();
    this.value = value;
    if (name) {
      this.name = name;
    }
  }

  get grad(): unknown {
    if (this.value && typeof this.value === 'object') {
      if ('derivative' in (this.value as ScalarLike)) {
        return (this.value as ScalarLike).derivative ?? null;
      }

      if ('grad' in (this.value as TensorLike)) {
        return (this.value as TensorLike).grad ?? null;
      }
    }

    return null;
  }

  update(next: T): void {
    this.value = next;
  }
}
