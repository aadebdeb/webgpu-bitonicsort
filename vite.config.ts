import { defineConfig } from 'vite';

import plainText from 'vite-plugin-plain-text';

export default defineConfig({
  base: process.env.GITHUB_PAGES ? '/webgpu-bitonicsort/' : '/',
  plugins: [
    plainText([/\.wgsl$/]),
  ],
});