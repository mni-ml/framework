const base = require('./jest.base.config.cjs');

/** @type {import('jest').Config} */
module.exports = {
  ...base,
  testMatch: [
    '<rootDir>/src/autodiff.test.ts',
    '<rootDir>/src/fast_ops.test.ts',
    '<rootDir>/src/module.test.ts',
    '<rootDir>/src/nn.test.ts',
    '<rootDir>/src/operators.test.ts',
    '<rootDir>/src/scalar.test.ts',
    '<rootDir>/src/tensor.test.ts',
    '<rootDir>/src/tensor_data.test.ts',
    '<rootDir>/src/tensor_functions.test.ts',
    '<rootDir>/src/tensor_ops.test.ts',
  ],
  moduleNameMapper: {
    '^\\./autodiff\\.js$': '<rootDir>/../tstorch/dist/autodiff.js',
    '^\\./datasets\\.js$': '<rootDir>/../tstorch/dist/datasets.js',
    '^\\./fast_ops\\.js$': '<rootDir>/../tstorch/dist/fast_ops.js',
    '^\\./fast_ops_worker\\.js$': '<rootDir>/../tstorch/dist/fast_ops_worker.js',
    '^\\./module\\.js$': '<rootDir>/test-shims/module.ts',
    '^\\./nn\\.js$': '<rootDir>/test-shims/nn.ts',
    '^\\./operators\\.js$': '<rootDir>/../tstorch/dist/operators.js',
    '^\\./optimizer\\.js$': '<rootDir>/test-shims/optimizer.ts',
    '^\\./scalar\\.js$': '<rootDir>/../tstorch/dist/scalar.js',
    '^\\./scalar_functions\\.js$': '<rootDir>/../tstorch/dist/scalar_functions.js',
    '^\\./tensor\\.js$': '<rootDir>/../tstorch/dist/tensor.js',
    '^\\./tensor_data\\.js$': '<rootDir>/../tstorch/dist/tensor_data.js',
    '^\\./tensor_functions\\.js$': '<rootDir>/../tstorch/dist/tensor_functions.js',
    '^\\./tensor_ops\\.js$': '<rootDir>/../tstorch/dist/tensor_ops.js',
  },
};
