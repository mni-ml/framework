import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve as pathResolve } from 'path';

export async function resolve(specifier, context, nextResolve) {
    // Remap relative .js imports to sibling .ts files when the .js file is absent.
    if (specifier.endsWith('.js') && (specifier.startsWith('./') || specifier.startsWith('../'))) {
        const parentPath = context.parentURL ? fileURLToPath(context.parentURL) : process.cwd();
        const parentDir = context.parentURL ? dirname(parentPath) : parentPath;
        const resolved = pathResolve(parentDir, specifier);

        // If .js doesn't exist, try .ts
        try {
            readFileSync(resolved);
        } catch {
            const tsPath = resolved.replace(/\.js$/, '.ts');
            try {
                readFileSync(tsPath);
                return nextResolve(specifier.replace(/\.js$/, '.ts'), context);
            } catch {
                // Fall through to default resolution
            }
        }
    }
    return nextResolve(specifier, context);
}
