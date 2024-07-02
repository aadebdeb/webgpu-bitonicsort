export class GraphRenderer {
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    this.context = canvas.getContext('2d') as CanvasRenderingContext2D
  }

  render(array: Float32Array) {
    this.context.fillStyle = 'white'
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height)

    const barWidth = this.canvas.width / array.length
    const barHeightScale = this.canvas.height / Math.max(...array)

    this.context.fillStyle = 'black'
    for (let i = 0; i < array.length; i++) {
      const x = i * barWidth
      const y = this.canvas.height - array[i] * barHeightScale
      const height = array[i] * barHeightScale
      this.context.fillRect(x, y, barWidth, height)
    }
  }
}