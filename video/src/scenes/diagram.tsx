import {makeScene2D, Rect, Img, Line, Layout} from '@motion-canvas/2d';
import {
  createRef,
  all,
  chain,
  waitFor,
  delay,
  easeInOutCubic,
  linear,
  ThreadGenerator,
} from '@motion-canvas/core';

import {Robot} from '../components/Robot';
import {Server, SERVER_H, SERVER_W} from '../components/Server';
import {BLUE, GREY, DARK, BG, withAlpha} from '../colors';
import ALabel from '../A_t.svg';
import OLabel from '../o_t.svg';

// =====================================================================
// Layout (all coords are view-centered).
// =====================================================================
const SERVER_POS: [number, number] = [829.5, 0];

const ROBOT_X = -1009;
const ROBOT_YS = [-600, -200, 200, 600];
const ROBOT_COLORS = [BLUE, GREY, GREY, GREY];
const ROBOT_LABELS = ['Robot 1', 'Robot 2', 'Robot 3', 'Robot 4'];
const ROBOT_MAX_SLOTS = [4, 7, 7, 7];
const NUM_ROBOTS = ROBOT_YS.length;

// Robot anchors (per-robot offsets, matching Robot.obsAnchor / actAnchor).
const OBS_ANCHOR_DY = -63.3;
const ACT_ANCHOR_DY = 58.7;
// Static obs/action arrows live in the gap between robots and server.
const ARROW_X_START = -278;
const ARROW_X_END = -63.5;

// Full server-box timeline. The left cursor is "now". Batches are scheduled
// as vertical columns of robot-aligned blocks and finish when the column fully
// exits through the cursor.
const SERVER_LEFT_X = SERVER_POS[0] - SERVER_W / 2;
const SERVER_RIGHT_X = SERVER_POS[0] + SERVER_W / 2;
const NOW_X = SERVER_LEFT_X + 54;
const TIMELINE_RIGHT_X = SERVER_RIGHT_X - 54;
const CLIP_CX = (NOW_X + TIMELINE_RIGHT_X) / 2;
const CLIP_W = TIMELINE_RIGHT_X - NOW_X;
const CLIP_H = SERVER_H - 88;
const BLOCK_H = 94;
const BATCH_GAP = 1;
const TICK_COUNT = 4;
const DOT_SIZE = 34;
const DOT_FLIGHT_DUR = 0.24;
const ARROW_LABEL_H = 54;

// Pixels per simulated second on the timeline.
const PPS = 260;

// =====================================================================
// Predefined schedule.
// =====================================================================
const T_OBS = 1.0;
const PHASES = [0.0, 0.40, 0.16, 0.62];
const SCHEDULE_GAP = 0.03;
const UNIT_BATCH_DUR = 0.84;
const BATCH_SIZE_RATIOS = [1, 1.5, 2, 2.25];

type ObsEvent = {t: number; robotIdx: number; batchIdx: number};
type Batch = {start: number; end: number; members: number[]};
type QueueEvent = {t: number; kind: 'consume' | 'replenish'};

const BATCH_MEMBERS: number[][] = [
  [0],
  [2, 0],
  [1, 3, 0],
  [2],
  [0, 1, 2, 3],
  [3, 1],
  [0, 2, 3],
  [1],
  [0, 1, 2],
  [3, 2, 0],
];

function makeSchedule(membersByBatch: number[][]): Batch[] {
  const batches: Batch[] = [];
  let start = 0;
  for (const members of membersByBatch) {
    const ratio = BATCH_SIZE_RATIOS[Math.min(members.length, 4) - 1];
    const dur = UNIT_BATCH_DUR * ratio;
    batches.push({start, end: start + dur, members});
    start += dur + SCHEDULE_GAP;
  }
  return batches;
}

const BATCHES = makeSchedule(BATCH_MEMBERS);
const T_TOTAL = Math.max(...BATCHES.map((b) => b.end)) + 0.8;

function observations(): ObsEvent[] {
  const obsEvents: ObsEvent[] = [];
  for (let i = 0; i < NUM_ROBOTS; i++) {
    for (let t = PHASES[i]; t < T_TOTAL; t += T_OBS) {
      obsEvents.push({t, robotIdx: i, batchIdx: -1});
    }
  }
  obsEvents.sort((a, b) => a.t - b.t);
  return obsEvents;
}

// =====================================================================
// Scene.
// =====================================================================
export default makeScene2D(function* (view) {
  view.size(3722, 1600);
  view.fill(BG);

  const obsEvents = observations();

  const robotRefs = ROBOT_YS.map(() => createRef<Robot>());
  const server = createRef<Server>();
  const scrollGroup = createRef<Layout>();

  view.add(
    <>
      {ROBOT_YS.map((y, i) => (
        <Robot
          ref={robotRefs[i]}
          label={ROBOT_LABELS[i]}
          color={ROBOT_COLORS[i]}
          labelBg={withAlpha(ROBOT_COLORS[i], 0.6)}
          maxSlots={ROBOT_MAX_SLOTS[i]}
          initialFill={ROBOT_MAX_SLOTS[i]}
          decay={0.18}
          position={[ROBOT_X, y]}
        />
      ))}
      <Server ref={server} position={SERVER_POS} />

      {/* Timeline ticks. */}
      {Array.from({length: TICK_COUNT + 1}, (_, i) => {
        const x = NOW_X + (CLIP_W * i) / TICK_COUNT;
        return (
          <Line
            points={[
              [x, -SERVER_H / 2 + 42],
              [x, SERVER_H / 2 - 42],
            ]}
            stroke={withAlpha(DARK, i === 0 ? 0.55 : 0.22)}
            lineWidth={2}
          />
        );
      })}

      {/* Timeline clip + scrolling content. */}
      <Rect
        position={[CLIP_CX, 0]}
        size={[CLIP_W, CLIP_H]}
        fill={null}
        clip
      >
        <Layout
          ref={scrollGroup}
          x={NOW_X - CLIP_CX}
          y={0}
          layout={false}
        />
      </Rect>

      {/* "Now" cursor at the left edge of the timeline. */}
      <Line
        points={[
          [NOW_X, -SERVER_H / 2 + 34],
          [NOW_X, SERVER_H / 2 - 34],
        ]}
        stroke={DARK}
        lineWidth={4}
      />

    </>,
  );

  for (const batch of BATCHES) {
    const width = Math.max(PPS * (batch.end - batch.start) - BATCH_GAP, 10);
    const group = new Layout({
      x: PPS * batch.start,
      y: 0,
      layout: false,
    });
    scrollGroup().add(group);

    for (const robotIdx of batch.members) {
      group.add(
        new Rect({
          x: 0,
          y: ROBOT_YS[robotIdx],
          offset: [-1, 0],
          width,
          height: BLOCK_H,
          fill: ROBOT_COLORS[robotIdx],
          radius: 3,
        }),
      );
    }
  }

  // Draw communication arrows after all base layers so they remain visible
  // above the server timeline.
  for (const y of ROBOT_YS) {
    view.add(
      new Line({
        points: [
          [ARROW_X_START, y + OBS_ANCHOR_DY],
          [ARROW_X_END, y + OBS_ANCHOR_DY],
        ],
        stroke: '#000000',
        lineWidth: 10,
        endArrow: true,
        arrowSize: 22,
      }),
    );
    view.add(
      new Line({
        points: [
          [ARROW_X_START, y + ACT_ANCHOR_DY],
          [ARROW_X_END, y + ACT_ANCHOR_DY],
        ],
        stroke: '#000000',
        lineWidth: 10,
        startArrow: true,
        arrowSize: 22,
      }),
    );
    view.add(
      new Img({
        position: [(ARROW_X_START + ARROW_X_END) / 2, y + OBS_ANCHOR_DY - 54],
        src: OLabel,
        height: ARROW_LABEL_H,
      }),
    );
    view.add(
      new Img({
        position: [(ARROW_X_START + ARROW_X_END) / 2, y + ACT_ANCHOR_DY + 54],
        src: ALabel,
        height: ARROW_LABEL_H,
      }),
    );
  }

  // -------------------------------------------------------------------
  // Animation primitives.
  //
  // Math: the scroll group's local x animates linearly from (NOW_X - CLIP_CX)
  // to (NOW_X - CLIP_CX - PPS * T_TOTAL). So at simulated time t, the
  // scroll-group view origin sits at view-x = NOW_X - PPS * t.
  //
  // A batch [t_s, t_s + D] is added at local x = PPS * t_s and has width
  // PPS * D. At t_s its left edge is on the "now" cursor; at t_s + D its
  // right edge reaches the cursor and the action chunks are sent to robots.
  // -------------------------------------------------------------------

  function* flyDot(
    robotIdx: number,
    y: number,
    direction: 'toServer' | 'toRobot',
  ): ThreadGenerator {
    const dot = new Rect({
      size: DOT_SIZE,
      radius: DOT_SIZE / 2,
      fill: ROBOT_COLORS[robotIdx],
      opacity: 0,
    });
    view.add(dot);

    const fromX = direction === 'toServer' ? ARROW_X_START : ARROW_X_END;
    const toX = direction === 'toServer' ? ARROW_X_END : ARROW_X_START;
    dot.position([fromX, y]);
    yield* dot.opacity(1, 0.04);
    yield* dot.position([toX, y], DOT_FLIGHT_DUR, easeInOutCubic);
    yield* dot.opacity(0, 0.04);
    dot.remove();
  }

  // -------------------------------------------------------------------
  // Master timeline.
  // -------------------------------------------------------------------
  // Hold briefly on the initial frame so the layout reads before motion.
  yield* waitFor(0.25);

  const animations: ThreadGenerator[] = [];
  const robotQueueEvents: QueueEvent[][] = ROBOT_YS.map(() => []);

  // Continuous scroll for the whole run.
  animations.push(
    scrollGroup().x(
      NOW_X - CLIP_CX - PPS * T_TOTAL,
      T_TOTAL,
      linear,
    ),
  );

  // Observation events send a lightweight colored dot along the static o_t arrow.
  for (const event of obsEvents) {
    robotQueueEvents[event.robotIdx].push({t: event.t, kind: 'consume'});
    animations.push(
      delay(
        event.t,
        flyDot(event.robotIdx, ROBOT_YS[event.robotIdx] + OBS_ANCHOR_DY, 'toServer'),
      ),
    );
  }

  // Batch lifecycle: scheduled batches scroll from the right; action
  // chunks send dots along A_t once each batch has fully exited through the cursor.
  for (const batch of BATCHES) {
    for (const robotIdx of batch.members) {
      robotQueueEvents[robotIdx].push({t: batch.end, kind: 'replenish'});
    }
    animations.push(
      delay(
        batch.end,
        all(
          ...batch.members.map((r) =>
            flyDot(r, ROBOT_YS[r] + ACT_ANCHOR_DY, 'toRobot'),
          ),
        ),
      ),
    );
  }

  for (let robotIdx = 0; robotIdx < robotQueueEvents.length; robotIdx++) {
    const events = robotQueueEvents[robotIdx].sort((a, b) => a.t - b.t);
    if (events.length === 0) continue;

    const ops: ThreadGenerator[] = [];
    let cursorT = 0;
    for (const event of events) {
      const duration = event.kind === 'consume' ? 0.28 : 0.45;
      ops.push(waitFor(Math.max(0, event.t - cursorT)));
      ops.push(
        event.kind === 'consume'
          ? robotRefs[robotIdx]().consumeAction(duration)
          : robotRefs[robotIdx]().receiveActionToFull(duration),
      );
      cursorT = Math.max(cursorT, event.t) + duration;
    }
    animations.push(chain(...ops));
  }

  yield* all(...animations);

  yield* waitFor(0.5);
});
