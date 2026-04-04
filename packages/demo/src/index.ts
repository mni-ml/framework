import runScalar from "./runScalar.js";
import runTensor from "./runTensor.js";
import runFastTensor from "./runFastTensor.js";

const mode = process.argv[2] ?? "tensor";

if (mode === "scalar") {
    runScalar();
} else if (mode === "fast") {
    runFastTensor();
} else {
    runTensor();
}
