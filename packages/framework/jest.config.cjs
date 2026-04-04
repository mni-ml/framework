/** @type {import('jest').Config} */
module.exports = {
  preset: 'ts-jest/presets/default-esm',
  testEnvironment: 'node',

  // Tell Jest that .ts files are ESM
  extensionsToTreatAsEsm: ['.ts'],

  // Transform TS using ts-jest in ESM mode
  transform: {
    '^.+\\.ts$': ['ts-jest', { useESM: true }],
  },

  // IMPORTANT:
  // When your TS imports "./x.js", map it to "./x" so Jest resolves "./x.ts"
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
};
