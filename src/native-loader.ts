/**
 * Node.js native addon loader.
 * Loads the platform-specific Rust N-API binary (CPU, CUDA, or WebGPU).
 */
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname_f = dirname(fileURLToPath(import.meta.url));

function hasCudaDriver(): boolean {
    if (process.platform === 'linux') {
        const searchPaths = [
            '/usr/lib/x86_64-linux-gnu',
            '/usr/lib64',
            '/usr/local/cuda/lib64',
            '/usr/local/cuda/targets/x86_64-linux/lib',
            '/usr/lib',
        ];
        const ldPath = process.env['LD_LIBRARY_PATH'] ?? '';
        if (ldPath) searchPaths.push(...ldPath.split(':').filter(Boolean));
        for (const dir of searchPaths) {
            if (existsSync(join(dir, 'libcuda.so')) || existsSync(join(dir, 'libcuda.so.1'))) {
                return true;
            }
        }
        return false;
    }
    if (process.platform === 'win32') {
        const system32 = join(process.env['SystemRoot'] ?? 'C:\\Windows', 'System32');
        if (existsSync(join(system32, 'nvcuda.dll'))) return true;
        const pathDirs = (process.env['PATH'] ?? '').split(';').filter(Boolean);
        for (const dir of pathDirs) {
            if (existsSync(join(dir, 'nvcuda.dll'))) return true;
        }
        return false;
    }
    return false;
}

function getPlatformPackages(): Record<string, string[]> {
    const cuda = hasCudaDriver();
    return {
        'darwin-arm64':  ['@mni-ml/framework-darwin-arm64-webgpu', '@mni-ml/framework-darwin-arm64'],
        'darwin-x64':    ['@mni-ml/framework-darwin-x64-webgpu',   '@mni-ml/framework-darwin-x64'],
        'linux-x64':     [
            ...(cuda ? ['@mni-ml/framework-linux-x64-gnu-cuda'] : []),
            '@mni-ml/framework-linux-x64-gnu-webgpu',
            '@mni-ml/framework-linux-x64-gnu',
        ],
        'linux-arm64':   ['@mni-ml/framework-linux-arm64-gnu'],
        'win32-x64':     [
            ...(cuda ? ['@mni-ml/framework-win32-x64-msvc-cuda'] : []),
            '@mni-ml/framework-win32-x64-msvc-webgpu',
            '@mni-ml/framework-win32-x64-msvc',
        ],
    };
}

function getLocalSuffixes(): Record<string, string[]> {
    const cuda = hasCudaDriver();
    return {
        'darwin-arm64': ['darwin-arm64-webgpu', 'darwin-arm64'],
        'darwin-x64':   ['darwin-x64-webgpu',   'darwin-x64'],
        'linux-x64':    [
            ...(cuda ? ['linux-x64-gnu-cuda'] : []),
            'linux-x64-gnu-webgpu',
            'linux-x64-gnu',
        ],
        'linux-arm64':  ['linux-arm64-gnu'],
        'win32-x64':    [
            ...(cuda ? ['win32-x64-msvc-cuda'] : []),
            'win32-x64-msvc-webgpu',
            'win32-x64-msvc',
        ],
    };
}

export function loadNative(): any {
    const require = createRequire(import.meta.url);
    const platform = process.platform;
    const arch = process.arch;
    const key = `${platform}-${arch}`;

    const candidates_pkg = getPlatformPackages()[key] ?? [];
    for (const pkgName of candidates_pkg) {
        try { return require(pkgName); } catch {}
    }

    const suffixes = getLocalSuffixes()[key] ?? [key];
    const ext = platform === 'win32' ? 'dll' : platform === 'darwin' ? 'dylib' : 'so';
    const candidates_file: string[] = [];
    for (const suffix of suffixes) {
        candidates_file.push(join(__dirname_f, '..', 'src', 'native', `mni-framework-native.${suffix}.node`));
    }
    candidates_file.push(join(__dirname_f, '..', 'src', 'native', 'target', 'release', `libmni_framework_native.${ext}`));
    for (const p of candidates_file) {
        if (existsSync(p)) {
            return require(p);
        }
    }

    const hint = candidates_pkg.length > 0
        ? `\n  Install prebuilt: npm install ${candidates_pkg[candidates_pkg.length - 1]}\n  For CUDA: npm install ${candidates_pkg[0]}\n  Or build from source: cd src/native && cargo build --release`
        : `\n  Build from source: cd src/native && cargo build --release`;
    throw new Error(
        `@mni-ml/framework: native addon not found for ${platform}-${arch}.${hint}`
    );
}

/**
 * Load the experimental cuTile Rust backend.
 *
 * This backend is a sibling of the default native addon — it is built from
 * `src/cutile` and exposes only a subset of the framework surface
 * (elementwise, activations, matmul, and plain tensor creation).  It is
 * intended for benchmarking and proof-of-concept work against NVIDIA's
 * cuTile Rust DSL, not as a drop-in replacement for `loadNative()`.
 *
 * Returns the raw N-API module so callers can opt-in op by op.
 */
export function loadCutile(): any {
    const require = createRequire(import.meta.url);
    const platform = process.platform;
    const arch = process.arch;
    const key = `${platform}-${arch}`;
    const suffixes = getLocalSuffixes()[key] ?? [key];
    const ext = platform === 'win32' ? 'dll' : platform === 'darwin' ? 'dylib' : 'so';

    const candidates_file: string[] = [];
    for (const suffix of suffixes) {
        candidates_file.push(
            join(__dirname_f, '..', 'src', 'cutile', `mni-framework-cutile.${suffix}.node`),
        );
    }
    candidates_file.push(
        join(__dirname_f, '..', 'src', 'cutile', 'target', 'release', `libmni_framework_cutile.${ext}`),
    );
    for (const p of candidates_file) {
        if (existsSync(p)) {
            return require(p);
        }
    }
    throw new Error(
        `@mni-ml/framework: cuTile addon not found for ${platform}-${arch}.\n  Build from source: cd src/cutile && cargo build --release --features napi`,
    );
}
