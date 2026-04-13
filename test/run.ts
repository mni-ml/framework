import { assert, startSuite, endSuite, summarize } from './helpers.js';

const suites = [
    ['test/tensor.test.ts', './tensor.test.js'],
    ['test/nn.test.ts', './nn.test.js'],
    ['test/autograd.test.ts', './autograd.test.js'],
    ['test/module.test.ts', './module.test.js'],
    ['test/native.test.ts', './native.test.js'],
    ['test/toy.test.ts', './toy.test.js'],
] as const;

function formatError(error: unknown): string {
    if (error instanceof Error) {
        return error.message;
    }
    return String(error);
}

async function runSuite(file: string, specifier: string): Promise<void> {
    startSuite(file);
    try {
        await import(specifier);
    } catch (error) {
        assert(false, `${file} failed to load: ${formatError(error)}`);
    } finally {
        endSuite(file);
    }
}

for (const [file, specifier] of suites) {
    await runSuite(file, specifier);
}

summarize();
