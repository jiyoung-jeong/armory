import {makeScene2D, Rect, Txt, Video, Layout, Circle} from '@motion-canvas/2d';
import {
  createRef,
  Reference,
  all,
  loop,
  delay,
  waitFor,
  easeInOutCubic,
} from '@motion-canvas/core';
import {BLUE, GREY, DARK, BG, withAlpha} from '../colors';

// Two side-by-side grids of the same 10 real-robot videos, one per scheduler.
// All videos start at t=0 and play together; around the midpoint we zoom each
// grid in on workstation11 (the fast robot) for a side-by-side comparison.

// workstation11 (the dynamic robot) is placed at the center-left cell of the
// grid (col 0, middle row). Robots are numbered 1-10 by grid position, not by
// the workstation index baked into the filenames.
const ORDER = [
  'workstation1',
  'workstation4',
  'workstation5',
  'workstation6',
  'workstation11',
  'workstation8',
  'workstation7',
  'workstation12',
  'workstation13',
  'workstation14',
];
const ZOOM_TARGET = 'workstation11';

const COLS = 2;
const ROWS = 5;
const CELL_W = 384;
const CELL_H = (CELL_W * 9) / 16; // 216, 16:9, matches the 1280x720 source
const GAP = 14;
const CELL_RADIUS = 14;

const GRID_W = COLS * CELL_W + (COLS - 1) * GAP;
const GRID_H = ROWS * CELL_H + (ROWS - 1) * GAP;

const FRAME_PAD = 12; // white inset between the grid and the rounded border
const FRAME_RADIUS = 28;
const FRAME_W = GRID_W + 2 * FRAME_PAD;
const FRAME_H = GRID_H + 2 * FRAME_PAD;

const MARGIN = 60; // outer breathing room so nothing is cut off
const LABEL_H = 100; // band above each grid for its label
const CENTER_GAP = 110; // horizontal spacing between the two grids

const VIEW_W = 2 * FRAME_W + CENTER_GAP + 2 * MARGIN;
const VIEW_H = MARGIN + LABEL_H + FRAME_H + MARGIN;
const GRIDS_CY = -VIEW_H / 2 + MARGIN + LABEL_H + FRAME_H / 2;
const LEFT_CX = -(FRAME_W + CENTER_GAP) / 2;
const RIGHT_CX = (FRAME_W + CENTER_GAP) / 2;

// Zoom scale so the target cell fills the grid (frame interior) width.
const ZOOM_S = GRID_W / CELL_W;
const FRAME_ZOOM_H = CELL_H * ZOOM_S + 2 * FRAME_PAD;
const LABEL_ORIG_Y = GRIDS_CY - FRAME_H / 2 - LABEL_H / 2;
const LABEL_ZOOM_Y = GRIDS_CY - FRAME_ZOOM_H / 2 - LABEL_H / 2;
const HEADER_Y = LABEL_ZOOM_Y - 135; // caption above the method labels when zoomed
const FOOTER_Y = GRIDS_CY + FRAME_ZOOM_H / 2 + 130; // caption below zoomed videos
const ZOOM_DUR = 1.5;

function cellPos(i: number): [number, number] {
  const col = i % COLS;
  const row = Math.floor(i / COLS);
  return [
    -GRID_W / 2 + CELL_W / 2 + col * (CELL_W + GAP),
    -GRID_H / 2 + CELL_H / 2 + row * (CELL_H + GAP),
  ];
}

const TARGET_POS = cellPos(ORDER.indexOf(ZOOM_TARGET));
// A non-target (quasi-static / slow) robot to highlight before workstation11.
// workstation6 sits at grid position "Robot #4".
const QUASI_TARGET = 'workstation6';
const QUASI_POS = cellPos(ORDER.indexOf(QUASI_TARGET));

interface Half {
  frame: Reference<Rect>;
  grid: Reference<Layout>;
  label: Reference<Txt>;
  videos: Reference<Video>[];
  initialText: string;
  zoomText: string;
}

// Zoom a grid onto a target cell and collapse its frame to that single video.
function* zoomInto(h: Half, pos: [number, number], dur: number) {
  yield* all(
    h.grid().scale(ZOOM_S, dur, easeInOutCubic),
    h.grid().position([-ZOOM_S * pos[0], -ZOOM_S * pos[1]], dur, easeInOutCubic),
    h.frame().size([FRAME_W, FRAME_ZOOM_H], dur, easeInOutCubic),
  );
}

// Restore a grid to the full-grid view.
function* zoomReset(h: Half, dur: number) {
  yield* all(
    h.grid().scale(1, dur, easeInOutCubic),
    h.grid().position([0, 0], dur, easeInOutCubic),
    h.frame().size([FRAME_W, FRAME_H], dur, easeInOutCubic),
  );
}

// A robot chip's "service beat": its status dot swells and settles on a fixed
// period for a stretch of `total` seconds, after an initial `phase` offset.
// Synchronized periods/phases read as orderly (homogeneous); varied ones read
// as chaotic (heterogeneous).
interface PulseSpec {
  dot: Reference<Circle>;
  period: number;
  phase: number;
}
function* pulseDot(spec: PulseSpec, total: number) {
  yield* waitFor(spec.phase);
  const cycles = Math.max(1, Math.round((total - spec.phase) / spec.period));
  yield* loop(cycles, function* () {
    yield* spec.dot().scale(1.55, spec.period * 0.45, easeInOutCubic);
    yield* spec.dot().scale(1, spec.period * 0.55, easeInOutCubic);
  });
}

// Emphasize the surviving grid's label: keep the "Lookahead" text but enlarge
// it and recolor it blue.
function* revealFinal(label: Txt) {
  yield* all(
    label.fontSize(96, 0.6, easeInOutCubic),
    label.fill(BLUE, 0.6, easeInOutCubic),
  );
}

export default makeScene2D(function* (view) {
  view.size(VIEW_W, VIEW_H);
  view.fill(BG);

  function buildHalf(
    centerX: number,
    base: string,
    labelText: string,
    zoomText: string,
    accent: string,
  ): Half {
    const frame = createRef<Rect>();
    const grid = createRef<Layout>();
    const label = createRef<Txt>();
    const videos = ORDER.map(() => createRef<Video>());

    view.add(
      <Rect
        ref={frame}
        position={[centerX, GRIDS_CY]}
        size={[FRAME_W, FRAME_H]}
        fill={'#FFFFFF'}
        stroke={accent}
        lineWidth={6}
        radius={FRAME_RADIUS}
        clip
      >
        <Layout ref={grid} layout={false}>
          {ORDER.map((ws, i) => (
            <Rect
              position={cellPos(i)}
              size={[CELL_W, CELL_H]}
              radius={CELL_RADIUS}
              fill={'#000000'}
              clip
            >
              <Video
                ref={videos[i]}
                src={`/${base}/${ws}.mp4`}
                size={[CELL_W, CELL_H]}
              />
              <Rect
                offset={[-1, -1]}
                position={[-CELL_W / 2 + 12, -CELL_H / 2 + 12]}
                fill={withAlpha(DARK, 0.62)}
                radius={9}
                layout
                padding={[6, 13]}
              >
                <Txt
                  text={`Robot #${i + 1}`}
                  fontFamily={'Helvetica Neue'}
                  fontWeight={600}
                  fontSize={24}
                  fill={'#FFFFFF'}
                />
              </Rect>
            </Rect>
          ))}
        </Layout>
      </Rect>,
    );
    // Label in the band above each grid.
    view.add(
      <Txt
        ref={label}
        position={[centerX, GRIDS_CY - FRAME_H / 2 - LABEL_H / 2]}
        text={labelText}
        fontFamily={'Helvetica Neue'}
        fontWeight={700}
        fontSize={60}
        fill={DARK}
      />,
    );
    return {frame, grid, label, videos, initialText: labelText, zoomText};
  }

  const halves: Half[] = [
    buildHalf(
      LEFT_CX,
      'maxbatch_proxy',
      'Earliest Deadline First',
      'Earliest Deadline First',
      GREY,
    ),
    buildHalf(
      RIGHT_CX,
      'lookahead_proxy',
      'Lookahead',
      'Lookahead',
      BLUE,
    ),
  ];

  // Overlay text: a centered intro statement and a top header for the zooms.
  // Captions are normal weight; key terms are bolded via nested Txt spans, so
  // the two zoom headers are separate rich-text nodes (faded in/out in turn).
  const intro = createRef<Txt>();
  const header1 = createRef<Txt>();
  const header2 = createRef<Txt>();
  view.add(
    <Txt
      ref={intro}
      position={[0, 0]}
      width={1500}
      fontFamily={'Helvetica Neue'}
      fontWeight={400}
      fontSize={72}
      fill={DARK}
      textAlign={'center'}
      textWrap
      opacity={0}
    >
      {'We introduce '}
      <Txt fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={72} fill={DARK}>
        Armory
      </Txt>
      {', a serving system that enables a single GPU to serve many robots with heterogeneity-aware, batched inference.'}
    </Txt>,
  );
  view.add(
    <Txt
      ref={header1}
      position={[0, HEADER_Y]}
      width={1720}
      fontFamily={'Helvetica Neue'}
      fontWeight={400}
      fontSize={48}
      fill={DARK}
      textAlign={'center'}
      textWrap
      opacity={0}
    >
      {'Some robots perform '}
      <Txt fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={48} fill={DARK}>
        quasi-static tasks
      </Txt>
      {' that don’t necessitate reactivity.'}
    </Txt>,
  );
  view.add(
    <Txt
      ref={header2}
      position={[0, HEADER_Y]}
      width={1720}
      fontFamily={'Helvetica Neue'}
      fontWeight={400}
      fontSize={48}
      fill={DARK}
      textAlign={'center'}
      textWrap
      opacity={0}
    >
      {'Other robots perform highly '}
      <Txt fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={48} fill={DARK}>
        dynamic, reactive tasks
      </Txt>
      {' that require better service in order to maintain throughput.'}
    </Txt>,
  );

  // Footer caption shown below the videos during the workstation11 zoom.
  const footer = createRef<Txt>();
  view.add(
    <Txt
      ref={footer}
      position={[0, FOOTER_Y]}
      width={1720}
      fontFamily={'Helvetica Neue'}
      fontWeight={400}
      fontSize={44}
      fill={DARK}
      textAlign={'center'}
      textWrap
      opacity={0}
    >
      {'Typically, '}
      <Txt fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={44} fill={DARK}>
        fast robots
      </Txt>
      {' perform '}
      <Txt fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={44} fill={DARK}>
        dynamic tasks
      </Txt>
      {', and '}
      <Txt fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={44} fill={DARK}>
        slow robots
      </Txt>
      {' perform '}
      <Txt fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={44} fill={DARK}>
        static tasks
      </Txt>
      {'.'}
    </Txt>,
  );

  // ---- Interstitial: homogeneous vs heterogeneous as the core problem ----
  // Two clusters of robot "chips". Left: identical robots whose service beats
  // pulse in unison (orderly). Right: a mix of fast (blue) and slow (gray)
  // robots beating at different rates (chaotic). Text-sparse; the contrast
  // carries the point.
  const problem = createRef<Layout>();
  const homoGroup = createRef<Layout>(); // homogeneous cluster (title + chips + caption)
  const hetGroup = createRef<Layout>(); // heterogeneous cluster
  const problemFooter = createRef<Txt>(); // centered footer revealed after both clusters
  const pulseSpecs: PulseSpec[] = [];
  const LX = -480; // resting x-center of the homogeneous cluster
  const RX = 480; // x-center of the heterogeneous cluster
  {
    const CHIP = 96;
    const CHIP_R = 20;
    const CGAP = 30;
    const COLS_P = 3;
    const ROWS_P = 2;
    const clusterW = COLS_P * CHIP + (COLS_P - 1) * CGAP;
    const clusterH = ROWS_P * CHIP + (ROWS_P - 1) * CGAP;
    const CY = -10;
    const TITLE_Y = CY - clusterH / 2 - 88;
    const CAP_Y = CY + clusterH / 2 + 150;
    const chipLocal = (i: number): [number, number] => {
      const c = i % COLS_P;
      const r = Math.floor(i / COLS_P);
      return [
        -clusterW / 2 + CHIP / 2 + c * (CHIP + CGAP),
        -clusterH / 2 + CHIP / 2 + r * (CHIP + CGAP),
      ];
    };

    view.add(<Layout ref={problem} layout={false} opacity={0} />);

    // Each cluster is a self-contained group (title + chips + caption) so it can
    // be positioned and revealed as one unit. Caption is built from rich-text
    // segments: [text, bold].
    const buildCluster = (
      groupRef: Reference<Layout>,
      x: number,
      title: string,
      specs: {color: string; period: number; phase: number}[],
      caption: [string, boolean][],
    ) => {
      problem().add(<Layout ref={groupRef} position={[x, 0]} layout={false} />);
      groupRef().add(
        <Txt
          position={[0, TITLE_Y]}
          text={title}
          fontFamily={'Helvetica Neue'}
          fontWeight={600}
          fontSize={46}
          fill={DARK}
        />,
      );
      specs.forEach((s, i) => {
        const [lx, ly] = chipLocal(i);
        const dot = createRef<Circle>();
        groupRef().add(
          <Rect
            position={[lx, CY + ly]}
            size={[CHIP, CHIP]}
            radius={CHIP_R}
            fill={withAlpha(s.color, 0.16)}
            stroke={withAlpha(s.color, 0.55)}
            lineWidth={3}
          >
            <Circle ref={dot} size={34} fill={s.color} />
          </Rect>,
        );
        pulseSpecs.push({dot, period: s.period, phase: s.phase});
      });
      groupRef().add(
        <Txt
          position={[0, CAP_Y]}
          width={760}
          fontFamily={'Helvetica Neue'}
          fontWeight={400}
          fontSize={46}
          fill={DARK}
          textAlign={'center'}
          textWrap
        >
          {caption.map(([t, b]) => (
            <Txt
              fontFamily={'Helvetica Neue'}
              fontWeight={b ? 700 : 400}
              fontSize={46}
              fill={DARK}
            >
              {t}
            </Txt>
          ))}
        </Txt>,
      );
    };

    // Homogeneous: all identical, all in phase.
    const homo = Array.from({length: 6}, () => ({
      color: BLUE,
      period: 0.95,
      phase: 0,
    }));
    // Heterogeneous: fast (blue, short period) and slow (gray, long period),
    // staggered so the beats never line up.
    const hetero = [
      {color: BLUE, period: 0.4, phase: 0.0},
      {color: GREY, period: 1.5, phase: 0.3},
      {color: BLUE, period: 0.4, phase: 0.15},
      {color: GREY, period: 1.5, phase: 0.55},
      {color: GREY, period: 1.4, phase: 0.2},
      {color: BLUE, period: 0.44, phase: 0.4},
    ];
    // Homogeneous starts centered (x=0); it slides to LX when the
    // heterogeneous cluster is revealed at RX.
    buildCluster(homoGroup, 0, 'Homogeneous', homo, [
      ['A homogeneous fleet is ', false],
      ['straightforward', true],
      [' to serve.', false],
    ]);
    buildCluster(hetGroup, RX, 'Heterogeneous', hetero, [
      ['A heterogeneous fleet is a much ', false],
      ['harder', true],
      [' scheduling problem.', false],
    ]);

    // Centered footer, revealed once both fleets are on screen.
    problem().add(
      <Txt
        ref={problemFooter}
        position={[0, CAP_Y + 150]}
        width={1560}
        fontFamily={'Helvetica Neue'}
        fontWeight={400}
        fontSize={46}
        fill={DARK}
        textAlign={'center'}
        textWrap
        opacity={0}
      >
        {'We build a '}
        <Txt fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={46} fill={DARK}>
          Lookahead
        </Txt>
        {' scheduler that can trade off between maximizing average service, or protecting specific classes of robots.'}
      </Txt>,
    );
  }

  // Full-screen overlay used to fade the whole scene out at the end.
  const fadeOverlay = createRef<Rect>();
  view.add(
    <Rect ref={fadeOverlay} size={[VIEW_W, VIEW_H]} fill={BG} opacity={0} />,
  );

  // ---- Intro: grids hidden, centered statement on screen ----
  for (const h of halves) {
    h.frame().scale(0);
    h.label().opacity(0);
  }
  yield* intro().opacity(1, 0.6, easeInOutCubic);
  yield* waitFor(3.6);
  yield* intro().opacity(0, 0.6, easeInOutCubic);

  // ---- Heterogeneity is the hard problem. Homogeneous fleet appears centered
  // first, then slides left as the heterogeneous fleet is revealed on the right.
  problem().scale(0.96);
  hetGroup().opacity(0);
  yield* all(
    problem().opacity(1, 0.6, easeInOutCubic),
    problem().scale(1, 0.6, easeInOutCubic),
  );
  yield* all(
    ...pulseSpecs.map((s) => pulseDot(s, 10.5)),
    // After the homogeneous fleet holds center stage, part the way for the
    // heterogeneous fleet.
    delay(
      2.4,
      all(
        homoGroup().position.x(LX, 0.8, easeInOutCubic),
        hetGroup().opacity(1, 0.6, easeInOutCubic),
      ),
    ),
    // Once both fleets are up, nudge them up to make room and reveal the
    // centered footer stating our answer.
    delay(
      5.2,
      all(
        problemFooter().opacity(1, 0.6, easeInOutCubic),
        homoGroup().position.y(-70, 0.6, easeInOutCubic),
        hetGroup().position.y(-70, 0.6, easeInOutCubic),
      ),
    ),
  );
  yield* waitFor(0.4);
  yield* problem().opacity(0, 0.6, easeInOutCubic);

  // ---- Grids expand; start playback together so they stay in lockstep ----
  for (const h of halves) {
    for (const v of h.videos) {
      v().play();
    }
  }
  yield* all(
    ...halves.flatMap((h) => [
      h.frame().scale(1, 1.0, easeInOutCubic),
      h.label().opacity(1, 0.9, easeInOutCubic),
    ]),
  );
  yield* waitFor(5.5);

  // ---- Zoom 1: a quasi-static (slow) robot. No method labels here; the
  // header fades in only after the zoom settles so it doesn't overlap the grid.
  yield* all(
    ...halves.map((h) => zoomInto(h, QUASI_POS, ZOOM_DUR)),
    ...halves.map((h) => h.label().opacity(0, 0.6, easeInOutCubic)),
  );
  yield* waitFor(0.5);
  yield* header1().opacity(1, 0.5, easeInOutCubic);
  yield* waitFor(4.5);

  // Zoom back out; restore the Naive / Smart labels over the full grids.
  yield* header1().opacity(0, 0.4, easeInOutCubic);
  yield* all(
    ...halves.map((h) => zoomReset(h, ZOOM_DUR)),
    ...halves.map((h) => h.label().opacity(1, 0.8, easeInOutCubic)),
  );
  yield* waitFor(2.0);

  // ---- Zoom 2: workstation11, the dynamic / latency-sensitive robot. The
  // method labels move up and transform; the header fades in after the zoom.
  yield* all(
    ...halves.flatMap((h) => [
      zoomInto(h, TARGET_POS, ZOOM_DUR),
      h.label().position.y(LABEL_ZOOM_Y, ZOOM_DUR, easeInOutCubic),
    ]),
  );
  yield* waitFor(0.5);
  yield* all(
    header2().opacity(1, 0.5, easeInOutCubic),
    footer().opacity(1, 0.5, easeInOutCubic),
  );
  yield* waitFor(10.0);

  // ---- Ending: the naive (gray) grid exits to the left while the smart
  // (blue) grid un-zooms, slides to center, and is revealed as "Armory". ----
  yield* all(
    header2().opacity(0, 0.4, easeInOutCubic),
    footer().opacity(0, 0.4, easeInOutCubic),
  );

  const gray = halves[0];
  const blue = halves[1];
  const OFF_LEFT = -(VIEW_W / 2 + FRAME_W);

  yield* all(
    // Naive grid un-zooms and slides off the left edge.
    zoomReset(gray, ZOOM_DUR),
    gray.frame().position.x(OFF_LEFT, ZOOM_DUR, easeInOutCubic),
    gray.label().position([OFF_LEFT, LABEL_ORIG_Y], ZOOM_DUR, easeInOutCubic),
    // Smart grid un-zooms and centers.
    zoomReset(blue, ZOOM_DUR),
    blue.frame().position.x(0, ZOOM_DUR, easeInOutCubic),
    blue.label().position([0, LABEL_ORIG_Y - 20], ZOOM_DUR, easeInOutCubic),
    revealFinal(blue.label()),
  );
  yield* waitFor(5.0);

  // ---- Fade the whole scene out over the final seconds ----
  yield* fadeOverlay().opacity(1, 2.5, easeInOutCubic);
});
