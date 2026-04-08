// Custom ESM loader: resolves .js imports to .ts files when the .js doesn't exist
import { access } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

export async function resolve(specifier, context, nextResolve) {
    // Only intercept relative imports ending in .js
    if (specifier.endsWith('.js') && (specifier.startsWith('./') || specifier.startsWith('../'))) {
        const parentDir = context.parentURL ? dirname(fileURLToPath(context.parentURL)) : process.cwd();
        const jsPath = join(parentDir, specifier);
        const tsPath = jsPath.replace(/\.js$/, '.ts');

        try {
            await access(jsPath);
            // .js exists, use it
            return nextResolve(specifier, context);
        } catch {
            try {
                await access(tsPath);
                // .ts exists, redirect
                return nextResolve(pathToFileURL(tsPath).href, context);
            } catch {
                // neither exists, let Node handle the error
                return nextResolve(specifier, context);
            }
        }
    }
    return nextResolve(specifier, context);
}
