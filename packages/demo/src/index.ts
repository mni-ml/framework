const mode = process.argv[2] ?? "tensor";

async function main(): Promise<void> {
    if (mode === "scalar") {
        const { default: runScalar } = await import("./runScalar.js");
        runScalar();
        return;
    }

    if (mode === "fast") {
        const { default: runFastTensor } = await import("./runFastTensor.js");
        runFastTensor();
        return;
    }

    if (mode === "mnist") {
        const { default: runMnist } = await import("./runMnist.js");
        await runMnist();
        return;
    }

    if (mode === "sentiment") {
        const { default: runSentiment } = await import("./runSentiment.js");
        runSentiment();
        return;
    }

    const { default: runTensor } = await import("./runTensor.js");
    runTensor();
}

try {
    await main();
} catch (error) {
    if (error instanceof Error && error.message.includes("Native addon not found")) {
        console.error(
            "The framework native addon is not built. Install Rust/Cargo, then run `pnpm --filter @mni-ml/framework run build:native`.",
        );
        process.exitCode = 1;
    } else {
        throw error;
    }
}
