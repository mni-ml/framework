declare module 'webgpu' {
    export function create(argv: string[]): GPU;
    export const globals: Record<string, unknown>;
}
