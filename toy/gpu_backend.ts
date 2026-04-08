import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);

let _device: GPUDevice | null = null;
let _gpu: GPU | null = null;

const pipelineCache = new Map<string, GPUComputePipeline>();

export async function getDevice(): Promise<GPUDevice> {
    if (_device) return _device;
    _gpu = create([]) as unknown as GPU;
    const adapter = await _gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter found');
    _device = await adapter.requestDevice();
    return _device;
}

export function destroyDevice(): void {
    _device?.destroy();
    _device = null;
    _gpu = null;
    pipelineCache.clear();
}

// Buffer helpers 

/**
 * Upload a Float64Array to the GPU as f32. Returns a STORAGE | COPY_SRC buffer.
 */
export function uploadBuffer(device: GPUDevice, data: Float64Array): GPUBuffer {
    const f32 = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
        f32[i] = Math.fround(data[i]!);
    }
    const buf = device.createBuffer({
        size: f32.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set(f32);
    buf.unmap();
    return buf;
}

/**
 * Create a GPU buffer for compute output (read-write storage, copyable for readback).
 */
export function createOutputBuffer(device: GPUDevice, count: number): GPUBuffer {
    return device.createBuffer({
        size: count * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
}

/**
 * Read a GPU buffer back to a Float64Array (f32 -> f64).
 */
export async function readbackBuffer(
    device: GPUDevice,
    srcBuffer: GPUBuffer,
    count: number,
): Promise<Float64Array> {
    const byteLen = count * Float32Array.BYTES_PER_ELEMENT;
    const staging = device.createBuffer({
        size: byteLen,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuffer, 0, staging, 0, byteLen);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const f32 = new Float32Array(staging.getMappedRange());
    const result = new Float64Array(count);
    for (let i = 0; i < count; i++) {
        result[i] = f32[i]!;
    }
    staging.unmap();
    staging.destroy();
    return result;
}

/**
 * Create a uniform buffer from an ArrayBuffer of packed metadata.
 * Only suitable for simple structs without array members (WGSL alignment rules).
 */
export function createUniformBuffer(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
    const aligned = Math.ceil(data.byteLength / 16) * 16;
    const buf = device.createBuffer({
        size: Math.max(aligned, 16),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange(0, data.byteLength)).set(new Uint8Array(data));
    buf.unmap();
    return buf;
}

/**
 * Create a read-only storage buffer for params that contain arrays.
 * Storage buffers don't have the 16-byte array element alignment requirement
 * that uniform buffers do in WGSL.
 */
export function createStorageParamsBuffer(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
    const buf = device.createBuffer({
        size: Math.max(data.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange(0, data.byteLength)).set(new Uint8Array(data));
    buf.unmap();
    return buf;
}

// Pipeline cache

export function getOrCreatePipeline(
    device: GPUDevice,
    shaderCode: string,
    entryPoint: string = 'main',
): GPUComputePipeline {
    const key = shaderCode;
    let pipeline = pipelineCache.get(key);
    if (pipeline) return pipeline;

    const module = device.createShaderModule({ code: shaderCode });
    pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint },
    });
    pipelineCache.set(key, pipeline);
    return pipeline;
}
