import {defineConfig} from 'vite';
import {createRequire} from 'module';

// Workaround: Node 24's ESM/CJS interop mishandles @motion-canvas/vite-plugin's
// default export. Pull it via createRequire so we get the function directly.
const require = createRequire(import.meta.url);
const motionCanvas = require('@motion-canvas/vite-plugin').default;
const ffmpeg = require('@motion-canvas/ffmpeg').default;

export default defineConfig({
  plugins: [
    motionCanvas({
      project: [
        './src/project.ts',
        './src/comparison.ts',
        './src/starvation.ts',
        './src/throughput_tradeoff.ts',
        './src/tier_lines.ts',
      ],
    }),
    ffmpeg(),
  ],
  // Allow serving the symlinked robot videos under public/ that resolve to the
  // shared data tree outside the project root.
  server: {fs: {allow: ['..', '/coc/flash7/rbansal66/vvla/data']}},
});
