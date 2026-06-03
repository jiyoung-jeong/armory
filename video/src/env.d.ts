/// <reference types="vite/client" />

declare module '*?scene' {
  const scene: import('@motion-canvas/core/lib/scenes/Scene').FullSceneDescription;
  export default scene;
}
