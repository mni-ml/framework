#!/usr/bin/env node
/**
 * Sync version across the main package and all platform packages.
 * Usage: node scripts/version.js 0.3.0
 */

import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');

const version = process.argv[2];
if (!version) {
    console.error('Usage: node scripts/version.js <version>');
    process.exit(1);
}

function updatePkg(path) {
    const pkg = JSON.parse(readFileSync(path, 'utf-8'));
    pkg.version = version;
    if (pkg.optionalDependencies) {
        for (const dep of Object.keys(pkg.optionalDependencies)) {
            pkg.optionalDependencies[dep] = version;
        }
    }
    writeFileSync(path, JSON.stringify(pkg, null, 2) + '\n');
    console.log(`  ${pkg.name} -> ${version}`);
}

console.log(`Updating all packages to v${version}:\n`);
updatePkg(join(root, 'package.json'));

const npmDir = join(root, 'npm');
for (const dir of readdirSync(npmDir)) {
    const pkgPath = join(npmDir, dir, 'package.json');
    try { updatePkg(pkgPath); } catch {}
}

console.log('\nDone. Commit and tag with:');
console.log(`  git commit -am "release: v${version}"`);
console.log(`  git tag v${version}`);
console.log(`  git push origin v${version}`);
