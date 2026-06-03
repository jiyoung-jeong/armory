import {Layout, LayoutProps, Rect} from '@motion-canvas/2d';
import {all, easeInOutCubic} from '@motion-canvas/core';

export interface ActionQueueProps extends LayoutProps {
  color: string;
  maxSlots: number;
  initialFill?: number;
  slotSize?: number;
  slotGap?: number;
  decay?: number;
  minOpacity?: number;
}

// A row of action "chunks" with a left→right opacity gradient.
// Leftmost = next-to-execute (full opacity); rightmost = newest/least certain (faded).
// consume(): leftmost slides off to the left and fades; remaining items shift left.
// replenish(): new chunk slides in from the right and fades up to its gradient opacity.
export class ActionQueue extends Layout {
  private readonly color: string;
  private readonly maxSlots: number;
  private readonly slotSize: number;
  private readonly slotPitch: number;
  private readonly decay: number;
  private readonly minOpacity: number;
  private readonly chunks: Rect[] = [];

  public constructor(props: ActionQueueProps) {
    super({layout: false, ...props});
    this.color = props.color;
    this.maxSlots = props.maxSlots;
    this.slotSize = props.slotSize ?? 60;
    const slotGap = props.slotGap ?? 29;
    this.slotPitch = this.slotSize + slotGap;
    this.decay = props.decay ?? 0.16;
    this.minOpacity = props.minOpacity ?? 0.08;

    const initialFill = Math.min(props.initialFill ?? this.maxSlots, this.maxSlots);
    for (let i = 0; i < initialFill; i++) {
      const chunk = new Rect({
        width: this.slotSize,
        height: this.slotSize,
        x: i * this.slotPitch,
        fill: this.color,
        opacity: this.opacityFor(i),
      });
      this.chunks.push(chunk);
      this.add(chunk);
    }
  }

  private opacityFor(slotIndex: number): number {
    return Math.max(this.minOpacity, 1 - slotIndex * this.decay);
  }

  public get fillCount(): number {
    return this.chunks.length;
  }

  public *consume(duration = 0.4) {
    if (this.chunks.length === 0) return;
    const removed = this.chunks.shift()!;
    yield* all(
      removed.opacity(0, duration, easeInOutCubic),
      removed.x(-this.slotPitch, duration, easeInOutCubic),
      ...this.chunks.map((chunk, i) =>
        all(
          chunk.x(i * this.slotPitch, duration, easeInOutCubic),
          chunk.opacity(this.opacityFor(i), duration, easeInOutCubic),
        ),
      ),
    );
    removed.remove();
  }

  public *replenish(duration = 0.4) {
    if (this.chunks.length >= this.maxSlots) return;
    const idx = this.chunks.length;
    const chunk = new Rect({
      width: this.slotSize,
      height: this.slotSize,
      x: (idx + 1) * this.slotPitch,
      fill: this.color,
      opacity: 0,
    });
    this.chunks.push(chunk);
    this.add(chunk);
    yield* all(
      chunk.x(idx * this.slotPitch, duration, easeInOutCubic),
      chunk.opacity(this.opacityFor(idx), duration, easeInOutCubic),
    );
  }

  // Refill the queue back to maxSlots in a single animated motion: all new
  // chunks slide in from the right together.
  public *replenishToFull(duration = 0.6) {
    const startCount = this.chunks.length;
    const toAdd = this.maxSlots - startCount;
    if (toAdd === 0) return;

    const newChunks: Rect[] = [];
    for (let i = 0; i < toAdd; i++) {
      const chunk = new Rect({
        width: this.slotSize,
        height: this.slotSize,
        x: (this.maxSlots + i + 1) * this.slotPitch,
        fill: this.color,
        opacity: 0,
      });
      this.chunks.push(chunk);
      newChunks.push(chunk);
      this.add(chunk);
    }

    yield* all(
      ...newChunks.map((chunk, j) => {
        const targetIdx = startCount + j;
        return all(
          chunk.x(targetIdx * this.slotPitch, duration, easeInOutCubic),
          chunk.opacity(this.opacityFor(targetIdx), duration, easeInOutCubic),
        );
      }),
    );
  }
}
