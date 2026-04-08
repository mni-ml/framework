import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

try {
    require.resolve('webgpu');
} catch {
    console.log('Skipping GPU tests: optional dependency `webgpu` is not installed.');
    process.exit(0);
}

const result = spawnSync(
    process.execPath,
    [
        '--experimental-vm-modules',
        '../../node_modules/jest/bin/jest.js',
        '--config',
        'jest.gpu.config.cjs',
        '--runInBand',
        '--passWithNoTests',
    ],
    {
        cwd: new URL('..', import.meta.url),
        stdio: 'inherit',
        env: process.env,
    },
);

if (result.error) {
    throw result.error;
}

process.exit(result.status ?? 1);
