import { startSuite, endSuite, summarize } from './helpers.js';

startSuite('test/tensor.test.ts');
await import('./tensor.test.js');
endSuite('test/tensor.test.ts');

startSuite('test/nn.test.ts');
await import('./nn.test.js');
endSuite('test/nn.test.ts');

startSuite('test/autograd.test.ts');
await import('./autograd.test.js');
endSuite('test/autograd.test.ts');

startSuite('test/module.test.ts');
await import('./module.test.js');
endSuite('test/module.test.ts');

startSuite('test/native.test.ts');
await import('./native.test.js');
endSuite('test/native.test.ts');

summarize();
