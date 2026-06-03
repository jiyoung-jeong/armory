import {Layout, LayoutProps, Rect, Txt, Line} from '@motion-canvas/2d';
import {BLUE, DARK, GREY} from '../colors';

export interface ServerProps extends LayoutProps {}

export const SERVER_W = 1699;
export const SERVER_H = 854;

// All positions below are server-local (origin at server center).
const ARMORY_CENTER: [number, number] = [87.5, -303.5];
const SUBTITLE_CENTER: [number, number] = [26.5, -161];

const STATE_MODULE_CENTER: [number, number] = [-384.5, 105];
const STATE_MODULE_SIZE: [number, number] = [676, 382];
const STATE_TITLE_CENTER: [number, number] = [-384.5, -12.5];
const STATE_CAPTION_CENTER: [number, number] = [-385, 341];

const BATCH_MODULE_CENTER: [number, number] = [387.5, 104.5];
const BATCH_MODULE_SIZE: [number, number] = [676, 383];
const BATCH_TITLE_CENTER: [number, number] = [385.5, -9];
const BATCH_CAPTION_CENTER: [number, number] = [378.5, 338];

// Slots in the batch grid (Figma absolute coords for each frame).
// Each is a 44x44 square. Server top-left in Figma is at (2023, 0), so server-local
// coords for a Figma point (fx, fy) are (fx - 2023 - SERVER_W/2, fy - SERVER_H/2).
const F2S_X = (fx: number) => fx - 2023 - SERVER_W / 2;
const F2S_Y = (fy: number) => fy - SERVER_H / 2;

const BATCH_SLOT_SIZE = 44;
type Slot = {x: number; y: number; color: string};
const BATCH_SLOTS: Slot[] = [
  // Top row (y=533)
  {x: F2S_X(3211 + 22), y: F2S_Y(533 + 22), color: BLUE},
  {x: F2S_X(3279 + 22), y: F2S_Y(533 + 22), color: BLUE},
  {x: F2S_X(3347 + 22), y: F2S_Y(533 + 22), color: BLUE},
  {x: F2S_X(3415 + 22), y: F2S_Y(532 + 22), color: BLUE},
  {x: F2S_X(3483 + 22), y: F2S_Y(533 + 22), color: BLUE},
  // Middle row (y=583)
  {x: F2S_X(3279 + 22), y: F2S_Y(583 + 22), color: GREY},
  {x: F2S_X(3415 + 22), y: F2S_Y(583 + 22), color: GREY},
  {x: F2S_X(3483 + 22), y: F2S_Y(583 + 22), color: GREY},
  // Bottom row (y=634)
  {x: F2S_X(3415 + 22), y: F2S_Y(634 + 22), color: GREY},
];

// Horizontal bar at the top of the grid (Vector 42).
const BATCH_BAR_CENTER: [number, number] = [
  F2S_X(3235.5 + 135),
  F2S_Y(492 + 12.75),
];

export class Server extends Layout {
  public constructor(props?: ServerProps) {
    super({size: [SERVER_W, SERVER_H], layout: false, ...props});

    // Outer panel.
    this.add(
      <Rect
        size={[SERVER_W, SERVER_H]}
        fill="#FFFFFF"
        stroke={DARK}
        lineWidth={8}
        radius={25}
      />,
    );

    // Armory title.
    this.add(
      <Txt
        position={ARMORY_CENTER}
        text="Armory"
        fontFamily={'Helvetica Neue'}
        fontWeight={700}
        fontSize={128}
        fill="#000000"
      />,
    );

    // Subtitle.
    this.add(
      <Txt
        position={SUBTITLE_CENTER}
        text="Schedule action chunks to maximize throughput"
        fontFamily={'Helvetica Neue'}
        fontWeight={700}
        fontSize={50}
        fill="#000000"
      />,
    );

    // State Tracking module.
    this.add(
      <Rect
        position={STATE_MODULE_CENTER}
        size={STATE_MODULE_SIZE}
        fill="#FFFFFF"
        stroke={DARK}
        lineWidth={8}
        radius={25}
      />,
    );
    this.add(
      <Txt
        position={STATE_TITLE_CENTER}
        text="State Tracking"
        fontFamily={'Helvetica Neue'}
        fontWeight={700}
        fontSize={64}
        fill="#000000"
      />,
    );
    this.add(
      <Txt
        position={STATE_CAPTION_CENTER}
        text="Mirror robot states"
        fontFamily={'Helvetica Neue'}
        fontWeight={400}
        fontSize={50}
        fill="#000000"
      />,
    );

    // Dynamic Batching module.
    this.add(
      <Rect
        position={BATCH_MODULE_CENTER}
        size={BATCH_MODULE_SIZE}
        fill="#FFFFFF"
        stroke={DARK}
        lineWidth={8}
        radius={25}
      />,
    );
    this.add(
      <Txt
        position={BATCH_TITLE_CENTER}
        text="Dynamic Batching"
        fontFamily={'Helvetica Neue'}
        fontWeight={700}
        fontSize={64}
        fill="#000000"
      />,
    );
    this.add(
      <Txt
        position={BATCH_CAPTION_CENTER}
        text="Batch on 1 remote GPU"
        fontFamily={'Helvetica Neue'}
        fontWeight={400}
        fontSize={50}
        fill="#000000"
      />,
    );

    // Batch grid: a horizontal bar over a 5x3 grid of small squares.
    this.add(
      <Rect
        position={BATCH_BAR_CENTER}
        size={[270, 25.5]}
        fill={null}
        stroke={DARK}
        lineWidth={4}
      />,
    );
    for (const slot of BATCH_SLOTS) {
      this.add(
        <Rect
          position={[slot.x, slot.y]}
          size={[BATCH_SLOT_SIZE, BATCH_SLOT_SIZE]}
          fill={slot.color}
        />,
      );
    }

    // GPU icon placeholder (a simple box).
    this.add(
      <Rect
        position={[F2S_X(2973 + 100.5), F2S_Y(523.13 + 60)]}
        size={[201, 120.12]}
        fill={null}
        stroke={DARK}
        lineWidth={6}
        radius={12}
      />,
    );
    this.add(
      <Txt
        position={[F2S_X(2973 + 100.5), F2S_Y(523.13 + 60)]}
        text="GPU"
        fontFamily={'Helvetica Neue'}
        fontWeight={700}
        fontSize={40}
        fill={DARK}
      />,
    );
  }
}
