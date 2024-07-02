import { plainText as shaderCode } from './compute.wgsl'
import { GraphRenderer } from './GraphRenderer';

const workgroupSize = 64

function sleep(milliseconds: number) {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

async function measurePerformance<T>(callback: () => Promise<T>): Promise<[T, number]> {
  const start = performance.now()
  const result = await callback()
  const end = performance.now()
  return [result, (end - start)]
}

// https://developer.mozilla.org/ja/docs/Web/JavaScript/Reference/Global_Objects/Math/log
function getBaseLog(x: number, y: number) {
  return Math.log(y) / Math.log(x);
}

async function bitonicsort(device: GPUDevice, array: Float32Array) {
  const base = Math.ceil(getBaseLog(2, array.length))
  const paddedArrayLength = 2 ** base
  const paddedArray = new Float32Array(Array.from({ length: paddedArrayLength }, (_, i) => i < array.length ? array[i] : 1))

  let readBuffer = device.createBuffer({
    size: paddedArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  })
  device.queue.writeBuffer(readBuffer, 0, paddedArray.buffer)
  let writeBuffer = device.createBuffer({
    size: paddedArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })
  const shaderModule = device.createShaderModule({
    code: shaderCode
  })
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'kernel'
    }
  })

  const uniformArray = new Uint32Array(2)
  const uniformBuffer = device.createBuffer({
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  })

  for (let i = 0; i < base; i++) {
    for (let j = 0; j <= i; j++) {
      uniformArray.set([i, j])
      device.queue.writeBuffer(uniformBuffer, 0, uniformArray)

      const commandEncoder = device.createCommandEncoder()
      const passEncoder = commandEncoder.beginComputePass()
      passEncoder.setPipeline(computePipeline)
      const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: readBuffer } },
          { binding: 1, resource: { buffer: writeBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } }
        ]
      })
      passEncoder.setBindGroup(0, bindGroup)
      passEncoder.dispatchWorkgroups(Math.ceil(paddedArray.length / workgroupSize))
      passEncoder.end()
      device.queue.submit([commandEncoder.finish()])

      ;[readBuffer, writeBuffer] = [writeBuffer, readBuffer]
    }
  }

  const resultBuffer = device.createBuffer({
    size: array.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  })
  const commandEncoder = device.createCommandEncoder()
  commandEncoder.copyBufferToBuffer(readBuffer, 0, resultBuffer, 0, array.byteLength)
  device.queue.submit([commandEncoder.finish()])

  await resultBuffer.mapAsync(GPUMapMode.READ, 0, array.byteLength)
  const resultArray = new Float32Array(resultBuffer.getMappedRange()).slice()
  resultBuffer.unmap()
  return resultArray
}

async function bitonicsortWithAnimation(device: GPUDevice, array: Float32Array, graphRenderer: GraphRenderer) {
  const base = Math.ceil(getBaseLog(2, array.length))
  const paddedArrayLength = 2 ** base
  const paddedArray = new Float32Array(Array.from({ length: paddedArrayLength }, (_, i) => i < array.length ? array[i] : 1))

  let readBuffer = device.createBuffer({
    size: paddedArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  })
  device.queue.writeBuffer(readBuffer, 0, paddedArray.buffer)
  let writeBuffer = device.createBuffer({
    size: paddedArray.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  })
  const shaderModule = device.createShaderModule({
    code: shaderCode
  })
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'kernel'
    }
  })

  const uniformArray = new Uint32Array(2)
  const uniformBuffer = device.createBuffer({
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  })

  for (let i = 0; i < base; i++) {
    for (let j = 0; j <= i; j++) {
      uniformArray.set([i, j])
      device.queue.writeBuffer(uniformBuffer, 0, uniformArray)

      const commandEncoder = device.createCommandEncoder()
      const passEncoder = commandEncoder.beginComputePass()
      passEncoder.setPipeline(computePipeline)
      const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: readBuffer } },
          { binding: 1, resource: { buffer: writeBuffer } },
          { binding: 2, resource: { buffer: uniformBuffer } }
        ]
      })
      passEncoder.setBindGroup(0, bindGroup)
      passEncoder.dispatchWorkgroups(Math.ceil(paddedArray.length / workgroupSize))
      passEncoder.end()
      device.queue.submit([commandEncoder.finish()])

      ;[readBuffer, writeBuffer] = [writeBuffer, readBuffer]

      const resultBuffer = device.createBuffer({
        size: array.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      })

      const copyCommandEncoder = device.createCommandEncoder()
      copyCommandEncoder.copyBufferToBuffer(readBuffer, 0, resultBuffer, 0, array.byteLength)
      device.queue.submit([copyCommandEncoder.finish()])
    
      await resultBuffer.mapAsync(GPUMapMode.READ, 0, array.byteLength)
      const resultArray = new Float32Array(resultBuffer.getMappedRange()).slice()
      graphRenderer.render(resultArray)
      resultBuffer.unmap()
      await sleep(200)
    }
  }

  const resultBuffer = device.createBuffer({
    size: array.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  })
  const commandEncoder = device.createCommandEncoder()
  commandEncoder.copyBufferToBuffer(readBuffer, 0, resultBuffer, 0, array.byteLength)
  device.queue.submit([commandEncoder.finish()])

  await resultBuffer.mapAsync(GPUMapMode.READ, 0, array.byteLength)
  const resultArray = new Float32Array(resultBuffer.getMappedRange()).slice()
  resultBuffer.unmap()
  return resultArray
}

const graphRenderer = new GraphRenderer(document.querySelector('canvas') as HTMLCanvasElement)
const dataLengthRange = document.querySelector('#data-length-range')! as HTMLInputElement
const measureResult = document.querySelector('#measure-result')!
const resetButton = document.querySelector('#reset-button') as HTMLButtonElement
const startAnimationButton = document.querySelector('#start-animation-button') as HTMLButtonElement
const startMeasureButton = document.querySelector('#start-measure-button') as HTMLButtonElement

function enableAllButtons() {
  resetButton.disabled = false
  startAnimationButton.disabled = false
  startMeasureButton.disabled = false
}
function disableAllButtons() {
  resetButton.disabled = true
  startAnimationButton.disabled = true
  startMeasureButton.disabled = true
}
function disableAllButtonsExceptReset() {
  resetButton.disabled = false
  startAnimationButton.disabled = true
  startMeasureButton.disabled = true
}

let array = new Float32Array(Array.from({ length: parseInt(dataLengthRange.value) }, () => Math.random()))
graphRenderer.render(array)

async function setup() {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) {
    throw new Error('WebGPU is not supported')
  }
  const device = await adapter.requestDevice()
  if (!device) {
    throw new Error('WebGPU is not supported')
  }

  resetButton.addEventListener('click', () => {
    measureResult.textContent = ''
    array = new Float32Array(Array.from({ length: parseInt(dataLengthRange.value) }, () => Math.random()))
    graphRenderer.render(array)
    enableAllButtons()
  })

  startAnimationButton.addEventListener('click', async () => {
    disableAllButtons()
    await bitonicsortWithAnimation(device, array, graphRenderer)
    disableAllButtonsExceptReset()
  })

  startMeasureButton.addEventListener('click', async () => {
    disableAllButtons()
    const [result, bitonicSortTime] = await measurePerformance(() => bitonicsort(device, array))
    const arraySortArray = array.slice()
    const [_arrayResult, arraySortTime] = await measurePerformance(async () => arraySortArray.sort())
    const measureResult = document.querySelector('#measure-result')!
    measureResult.textContent = `Bitonic sort: ${bitonicSortTime}ms, Array#sort: ${arraySortTime}ms`
    graphRenderer.render(result)
    disableAllButtonsExceptReset()


  })
}

setup()