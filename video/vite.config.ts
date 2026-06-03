import {defineConfig} from 'vite';
import {createRequire} from 'module';

// Workaround: Node 24's ESM/CJS interop mishandles @motion-canvas/vite-plugin's
// default export. Pull it via createRequire so we get the function directly.
const require = createRequire(import.meta.url);
const motionCanvas = require('@motion-canvas/vite-plugin').default;

export default defineConfig({
  plugins: [motionCanvas()],
});
