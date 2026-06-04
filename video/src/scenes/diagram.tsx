import {makeScene2D, Rect, Img, Line, Layout, Txt} from '@motion-canvas/2d';
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
import {BLUE, GREY, DARK, BG, YELLOW, withAlpha} from '../colors';
import ALabel from '../A_t.svg';
import OLabel from '../o_t.svg';

// =====================================================================
// Layout (all coords are view-centered).
// =====================================================================
const DIAGRAM_Y = 140;
const SERVER_POS: [number, number] = [829.5, DIAGRAM_Y];

const ROBOT_X = -1009;
const SERVER_ROW_YS = [-490, -163, 163, 490];
const ROBOT_YS = SERVER_ROW_YS.map((y) => y + DIAGRAM_Y);
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
const SECTION_LABEL_Y = -630;
const SECTION_LABEL_FONT_SIZE = 104;
const CAPTION_Y = -760;
const CAPTION_W = 3150;
const CAPTION_FONT_SIZE = 74;
const CAPTION_LEAD_IN_DUR = 3;
const CAPTION_SLOT_DUR = 5;
const CAPTION_TAIL_DUR = 5;

// Pixels per simulated second on the timeline.
const PPS = 260;

// =====================================================================
// Predefined schedule.
// =====================================================================
const T_OBS = 1.0;
const PHASES = [0.0, 0.40, 0.16, 0.62];
const SCHEDULE_GAP = 0.03;
const UNIT_BATCH_DUR = 1;
const BATCH_SIZE_RATIOS = [1, 1.5, 2, 2.25];

type ObsEvent = {t: number; robotIdx: number; batchIdx: number};
type Batch = {start: number; end: number; members: number[]};
type QueueEvent = {t: number; kind: 'consume' | 'replenish'};

const BATCH_MEMBERS: number[][] = [
  [0],
  [0, 2],
  [0, 1, 3],
  [0],
  [0, 1, 2, 3],
  [0, 3],
  [0, 2],
  [0, 1],
  [0, 1, 2],
  [0, 1, 2, 3],
  [0],
  [0, 2, 3],
  [0, 1],
  [0, 2],
  [0, 1, 3],
  [1],
  [0, 1, 2],
  [0, 3],
  [0, 2],
  [0, 1],
  [0, 1, 2, 3],
  [0],
  [0, 2],
  [0, 1, 3],
  [0, 3],
  [0, 1],
  [2],
  [0, 1, 2],
  [0, 3],
  [0, 2],
  [0, 1],
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
// Index of the full (4-robot) batch spotlighted during the "larger batches"
// caption. Its cursor passage (~14.5-16.8s) lands inside that caption window.
const HIGHLIGHT_BATCH_IDX = 9;
// Captions as [text, bold] segments so key terms can be emphasized.
const CAPTIONS: [string, boolean][][] = [
  [['Robots continuously consume actions and send observations to the server.', false]],
  [
    ['Robots with shorter execution horizons require more frequent inferences. We refer to them as ', false],
    ['fast robots', true],
    ['.', false],
  ],
  [['Larger batches have higher throughput, but also higher latency, costing robot reactivity.', false]],
  [
    ['Effective serving requires scheduling ', false],
    ['which robots to serve at the right times', true],
    ['.', false],
  ],
];

// Build the inline Txt spans for a caption, normal weight except bold segments.
function captionSpans(segments: [string, boolean][]): Txt[] {
  return segments.map(
    ([t, b]) =>
      new Txt({
        text: t,
        fontFamily: 'Helvetica Neue',
        fontWeight: b ? 700 : 400,
        fontSize: CAPTION_FONT_SIZE,
        fill: '#000000',
      }),
  );
}
const T_RENDER =
  CAPTION_LEAD_IN_DUR + CAPTIONS.length * CAPTION_SLOT_DUR + CAPTION_TAIL_DUR;

function observations(): ObsEvent[] {
  const obsEvents: ObsEvent[] = [];
  for (let i = 0; i < NUM_ROBOTS; i++) {
    for (let t = PHASES[i]; t < T_RENDER; t += T_OBS) {
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
  const overlay = createRef<Layout>();
  const caption = createRef<Txt>();

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
              [x, DIAGRAM_Y - SERVER_H / 2 + 42],
              [x, DIAGRAM_Y + SERVER_H / 2 - 42],
            ]}
            stroke={withAlpha(DARK, i === 0 ? 0.55 : 0.22)}
            lineWidth={2}
          />
        );
      })}

      {/* Timeline clip + scrolling content. */}
      <Rect
        position={[CLIP_CX, DIAGRAM_Y]}
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
          [NOW_X, DIAGRAM_Y - SERVER_H / 2 + 34],
          [NOW_X, DIAGRAM_Y + SERVER_H / 2 - 34],
        ]}
        stroke={DARK}
        lineWidth={4}
      />

    </>,
  );

  // Spotlight outline for the highlighted batch (filled in below, pulsed later).
  let highlightBatchOutline: Rect | null = null;

  for (let batchIdx = 0; batchIdx < BATCHES.length; batchIdx++) {
    const batch = BATCHES[batchIdx];
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
          y: SERVER_ROW_YS[robotIdx],
          offset: [-1, 0],
          width,
          height: BLOCK_H,
          fill: ROBOT_COLORS[robotIdx],
          radius: 3,
        }),
      );
    }

    if (batchIdx === HIGHLIGHT_BATCH_IDX) {
      const ys = batch.members.map((r) => SERVER_ROW_YS[r]);
      const yTop = Math.min(...ys) - BLOCK_H / 2;
      const yBot = Math.max(...ys) + BLOCK_H / 2;
      highlightBatchOutline = new Rect({
        x: width / 2,
        y: (yTop + yBot) / 2,
        size: [width + 22, yBot - yTop + 22],
        fill: null,
        stroke: YELLOW,
        lineWidth: 12,
        radius: 12,
        shadowColor: YELLOW,
        shadowBlur: 0,
        opacity: 0,
      });
      group.add(highlightBatchOutline);
    }
  }

  // Highlight box behind the obs/action arrows (the streaming "middle section"),
  // pulsed during the first caption. Sits in the gap between robots and server.
  const CHANNEL_CX = (ARROW_X_START + ARROW_X_END) / 2;
  const channelTop = ROBOT_YS[0] + OBS_ANCHOR_DY - ARROW_LABEL_H - 40;
  const channelBot =
    ROBOT_YS[ROBOT_YS.length - 1] + ACT_ANCHOR_DY + ARROW_LABEL_H + 40;
  const channelHighlight = new Rect({
    x: CHANNEL_CX,
    y: (channelTop + channelBot) / 2,
    size: [ARROW_X_END - ARROW_X_START + 52, channelBot - channelTop],
    fill: withAlpha(YELLOW, 0.12),
    stroke: withAlpha(YELLOW, 0.55),
    lineWidth: 6,
    radius: 30,
    shadowColor: YELLOW,
    shadowBlur: 0,
    opacity: 0,
  });
  view.add(channelHighlight);

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

  view.add(
    <Layout ref={overlay} layout={false}>
      <Txt
        position={[ROBOT_X, SECTION_LABEL_Y]}
        text={'Robot'}
        fontFamily={'Helvetica Neue'}
        fontWeight={400}
        fontSize={SECTION_LABEL_FONT_SIZE}
        fill="#000000"
      />
      <Txt
        position={[(NOW_X + TIMELINE_RIGHT_X) / 2, SECTION_LABEL_Y]}
        text={'Server'}
        fontFamily={'Helvetica Neue'}
        fontWeight={400}
        fontSize={SECTION_LABEL_FONT_SIZE}
        fill="#000000"
      />
      <Txt
        ref={caption}
        position={[0, CAPTION_Y]}
        width={CAPTION_W}
        fontFamily={'Helvetica Neue'}
        fontWeight={400}
        fontSize={CAPTION_FONT_SIZE}
        fill="#000000"
        textAlign={'center'}
        opacity={0}
      >
        {captionSpans(CAPTIONS[0])}
      </Txt>
    </Layout>,
  );

  // -------------------------------------------------------------------
  // Animation primitives.
  //
  // Math: the scroll group's local x animates linearly from (NOW_X - CLIP_CX)
  // to (NOW_X - CLIP_CX - PPS * T_RENDER). So at simulated time t, the
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
    overlay().moveToTop();

    const fromX = direction === 'toServer' ? ARROW_X_START : ARROW_X_END;
    const toX = direction === 'toServer' ? ARROW_X_END : ARROW_X_START;
    dot.position([fromX, y]);
    yield* dot.opacity(1, 0.04);
    yield* dot.position([toX, y], DOT_FLIGHT_DUR, easeInOutCubic);
    yield* dot.opacity(0, 0.04);
    dot.remove();
  }

  // Fade a glowing highlight node in, hold, then out.
  function* spotlight(node: Rect, hold: number, glow = 44): ThreadGenerator {
    yield* all(
      node.opacity(1, 0.3, easeInOutCubic),
      node.shadowBlur(glow, 0.3, easeInOutCubic),
    );
    yield* waitFor(hold);
    yield* all(
      node.opacity(0, 0.3, easeInOutCubic),
      node.shadowBlur(0, 0.3, easeInOutCubic),
    );
  }

  function* showCaptions(): ThreadGenerator {
    yield* waitFor(CAPTION_LEAD_IN_DUR);
    for (let i = 0; i < CAPTIONS.length; i++) {
      caption().removeChildren();
      caption().add(captionSpans(CAPTIONS[i]));
      yield* caption().opacity(1, 0.28, easeInOutCubic);
      const hold = CAPTION_SLOT_DUR - 0.72;
      if (i === 1) {
        // "Some robots have shorter execution horizons..." — spotlight Robot 1
        // (the short-horizon robot) for the duration of this caption.
        yield* all(waitFor(hold), robotRefs[0]().highlight(hold));
      } else {
        yield* waitFor(hold);
      }
      yield* caption().opacity(0, 0.28, easeInOutCubic);
      yield* waitFor(0.16);
    }
    yield* waitFor(CAPTION_TAIL_DUR);
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
      NOW_X - CLIP_CX - PPS * T_RENDER,
      T_RENDER,
      linear,
    ),
  );
  animations.push(showCaptions());

  // Spotlight the obs/action streaming channel during the first caption
  // ("Robots continuously consume actions and send observations...", ~3.0-7.84s).
  animations.push(delay(3.0, spotlight(channelHighlight, 4.0, 26)));

  // Spotlight the full batch (idx 9) as it scrolls toward the cursor, lining up
  // with the start of the "larger batches have higher throughput, but also
  // higher latency" caption (text appears at ~13.0s, batch exits cursor ~16.8s).
  if (highlightBatchOutline) {
    animations.push(delay(13.1, spotlight(highlightBatchOutline, 2.4)));
  }

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
    if (batch.end > T_RENDER) continue;

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
});
