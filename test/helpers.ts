const green = '\x1b[32m';
const red = '\x1b[31m';
const yellow = '\x1b[33m';
const bold = '\x1b[1m';
const dim = '\x1b[2m';
const reset = '\x1b[0m';
const greenBg = '\x1b[30m\x1b[42m';
const redBg = '\x1b[37m\x1b[41m';

let totalPassed = 0;
let totalFailed = 0;
let totalSkipped = 0;
let suitePassed = 0;
let suiteFailed = 0;
let suiteSkipped = 0;
let suitesRun = 0;
let suitesFailed = 0;
const failures: string[] = [];
const startTime = Date.now();

export function assert(cond: boolean, msg: string): void {
    if (!cond) {
        suiteFailed++;
        totalFailed++;
        failures.push(msg);
        console.log(`    ${red}\u2716${reset} ${msg}`);
    } else {
        suitePassed++;
        totalPassed++;
        console.log(`    ${green}\u2713${reset} ${dim}${msg}${reset}`);
    }
}

export function assertClose(a: number, b: number, tol: number = 1e-4, msg: string = ''): void {
    if (Math.abs(a - b) > tol) {
        suiteFailed++;
        totalFailed++;
        const detail = `${msg}: ${a} != ${b} (tol=${tol})`;
        failures.push(detail);
        console.log(`    ${red}\u2716${reset} ${detail}`);
    } else {
        suitePassed++;
        totalPassed++;
        console.log(`    ${green}\u2713${reset} ${dim}${msg}${reset}`);
    }
}

export function skip(msg: string): void {
    suiteSkipped++;
    totalSkipped++;
    console.log(`    ${yellow}\u25CB${reset} ${dim}skipped: ${msg}${reset}`);
}

export function section(name: string): void {
    console.log(`  ${name}`);
}

export function startSuite(file: string): void {
    suitesRun++;
    suitePassed = 0;
    suiteFailed = 0;
    suiteSkipped = 0;
    console.log('');
}

export function endSuite(file: string): void {
    const badge = suiteFailed > 0
        ? `${redBg} FAIL ${reset}`
        : `${greenBg} PASS ${reset}`;
    console.log(`${badge} ${file}`);
    if (suiteFailed > 0) suitesFailed++;
}

export function summarize(): void {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(3);
    const total = totalPassed + totalFailed + totalSkipped;

    console.log('');

    // Suites line
    const suitesPassedCount = suitesRun - suitesFailed;
    if (suitesFailed > 0) {
        console.log(`${bold}Test Suites:${reset} ${red}${suitesFailed} failed${reset}, ${green}${suitesPassedCount} passed${reset}, ${suitesRun} total`);
    } else {
        console.log(`${bold}Test Suites:${reset} ${green}${suitesPassedCount} passed${reset}, ${suitesRun} total`);
    }

    // Tests line
    const parts: string[] = [];
    if (totalFailed > 0) parts.push(`${red}${totalFailed} failed${reset}`);
    if (totalSkipped > 0) parts.push(`${yellow}${totalSkipped} skipped${reset}`);
    parts.push(`${green}${totalPassed} passed${reset}`);
    console.log(`${bold}Tests:${reset}       ${parts.join(', ')}, ${total} total`);

    // Time
    console.log(`${bold}Time:${reset}        ${elapsed} s`);

    if (totalFailed > 0) {
        console.log(`\n${red}Failures:${reset}`);
        for (const f of failures) console.log(`  ${red}\u2716${reset} ${f}`);
        process.exit(1);
    }
}
