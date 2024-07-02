struct Params {
  blockStep: u32,
  subBlockStep: u32
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let blockStep = params.blockStep;
  let subBlockStep = params.subBlockStep;
  let d = 1u << (blockStep - subBlockStep);
  var asc = ((index >> blockStep) & 2) == 0;
  var targetIndex = 0u;
  if (index & d) == 0 {
    targetIndex = index | d;
  } else {
    targetIndex = index & ~d;
    asc = !asc;
  }
  let v0 = input[index];
  let v1 = input[targetIndex];
  output[index] = select(v0, v1, (v0 > v1) == asc);
}